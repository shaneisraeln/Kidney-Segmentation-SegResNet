import os
import time
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from monai.inferers import sliding_window_inference
from src.utils.logger import get_logger
from src.training.losses import build_loss
from src.training.metrics import SegmentationMetrics
from src.models import get_model

logger = get_logger("Trainer")

class Trainer:
    """
    Robust Trainer for 2D/3D Kidney and Tumor Segmentation models.
    Supports Class-Weighted DiceCE Loss, AMP, Gradient Accumulation, and Early Stopping.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        training_cfg = config.get("training", {})
        self.epochs = training_cfg.get("epochs", 30)
        self.lr = float(training_cfg.get("learning_rate", 3e-4))
        self.weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        self.accum_steps = training_cfg.get("gradient_accumulation_steps", 1)
        self.patience = training_cfg.get("early_stopping_patience", 15)
        self.use_amp = training_cfg.get("amp", True) and torch.cuda.is_available()
        self.checkpoint_dir = training_cfg.get("checkpoint_dir", "checkpoints")
        self.patch_size = tuple(training_cfg.get("patch_size", [48, 64, 64]))
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Loss Function with Configurable Class Weights
        loss_cfg = config.get("loss", {})
        loss_name = loss_cfg.get("name", "dice_ce")
        class_weights = loss_cfg.get("weights", [0.2, 1.0, 3.0, 3.0])
        num_classes = config.get("model", {}).get("out_channels", 4)
        
        self.criterion = build_loss(loss_name, num_classes=num_classes, class_weights=class_weights)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.metrics = SegmentationMetrics(num_classes=num_classes)
        
        self.best_val_dice = -1.0
        self.start_epoch = 0
        self.no_improve_count = 0

    def save_checkpoint(self, epoch: int, val_dice: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_dice": self.best_val_dice,
            "config": self.config
        }
        
        last_path = os.path.join(self.checkpoint_dir, "last_model.pth")
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)
            logger.info(f"Saved new best model checkpoint to {best_path} with Val Dice: {val_dice:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"]:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_val_dice = checkpoint.get("best_val_dice", -1.0)
        logger.info(f"Resumed training from epoch {self.start_epoch} with Best Val Dice: {self.best_val_dice:.4f}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_dice_loss = 0.0
        total_ce_loss = 0.0
        self.optimizer.zero_grad()
        
        start_time = time.time()
        for step, batch in enumerate(self.train_loader):
            if isinstance(batch, list):
                images = torch.cat([b["image"] for b in batch], dim=0).to(self.device)
                labels = torch.cat([b["label"] for b in batch], dim=0).to(self.device)
            else:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images)
                loss, d_loss, ce_loss = self.criterion(logits, labels)
                loss_scaled = loss / self.accum_steps

            self.scaler.scale(loss_scaled).backward()
            
            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            total_dice_loss += d_loss.item()
            total_ce_loss += ce_loss.item()

        n_steps = len(self.train_loader)
        epoch_loss = total_loss / n_steps
        epoch_d_loss = total_dice_loss / n_steps
        epoch_ce_loss = total_ce_loss / n_steps
        elapsed = time.time() - start_time
        
        curr_lr = self.optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch [{epoch}/{self.epochs}] Train Loss: {epoch_loss:.4f} "
            f"(Dice Loss: {epoch_d_loss:.4f}, CE Loss: {epoch_ce_loss:.4f}) | "
            f"LR: {curr_lr:.6f} | ({elapsed:.1f}s)"
        )
        return {
            "loss": epoch_loss,
            "dice_loss": epoch_d_loss,
            "ce_loss": epoch_ce_loss
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        val_loss = 0.0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Full-volume sliding window inference
            logits = sliding_window_inference(
                inputs=images,
                roi_size=self.patch_size,
                sw_batch_size=2,
                predictor=self.model,
                overlap=0.5
            )
            loss, _, _ = self.criterion(logits, labels)
                
            val_loss += loss.item()
            self.metrics.update(logits, labels)

        avg_val_loss = val_loss / len(self.val_loader)
        metric_results = self.metrics.compute()
        metric_results["val_loss"] = avg_val_loss
        
        logger.info(
            f"Epoch [{epoch}/{self.epochs}] Val Loss: {avg_val_loss:.4f} | "
            f"Val Mean Dice: {metric_results.get('mean_dice', 0.0):.4f} | "
            f"Kidney: {metric_results.get('dice_kidney', 0.0):.4f} | "
            f"Tumor: {metric_results.get('dice_tumor', 0.0):.4f} | "
            f"Cyst: {metric_results.get('dice_cyst', 0.0):.4f}"
        )
        return metric_results

    def fit(self, resume_path: Optional[str] = None):
        if resume_path:
            self.load_checkpoint(resume_path)
            
        logger.info(f"Starting enhanced training for {self.epochs} epochs on device: {self.device}")
        
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            val_dice = val_metrics.get("mean_dice", 0.0)
            self.scheduler.step()

            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.no_improve_count = 0
                self.save_checkpoint(epoch, val_dice, is_best=True)
            else:
                self.no_improve_count += 1
                self.save_checkpoint(epoch, val_dice, is_best=False)

            if self.no_improve_count >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement.")
                break

        logger.info(f"Training completed. Best Validation Dice: {self.best_val_dice:.4f}")
