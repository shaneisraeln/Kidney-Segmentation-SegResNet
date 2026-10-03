import os
import sys
import time
import argparse
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.logger import get_logger
from src.data.dataset import create_dataloaders
from src.models import get_model
from src.training.trainer import Trainer

logger = get_logger("TrainScript")

def main():
    parser = argparse.ArgumentParser(description="Train Kidney Segmentation Models")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    dataset_cfg = config.get("dataset", {})
    data_dir = dataset_cfg.get("data_dir", "kits21/data")
    split_dir = dataset_cfg.get("split_dir", "data/splits")

    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 2)
    patch_size = tuple(training_cfg.get("patch_size", [96, 96, 96]))

    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=batch_size,
        patch_size=patch_size
    )

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "unet3d")
    in_channels = model_cfg.get("in_channels", 1)
    out_channels = model_cfg.get("out_channels", 4)

    logger.info(f"Instantiating model: {model_name}")
    model = get_model(model_name, in_channels=in_channels, out_channels=out_channels)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    start_time = time.time()
    trainer.fit(resume_path=args.resume)
    elapsed_minutes = (time.time() - start_time) / 60.0

    # Record in experiment log
    exp_log_path = "experiments/experiment_log.csv"
    os.makedirs(os.path.dirname(exp_log_path), exist_ok=True)
    exp_entry = {
        "experiment_id": f"{model_name}_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "loss": config.get("loss", {}).get("name", "dice"),
        "learning_rate": training_cfg.get("learning_rate"),
        "batch_size": batch_size,
        "patch_size": str(patch_size),
        "best_val_dice": trainer.best_val_dice,
        "training_time_min": round(elapsed_minutes, 2),
        "device": str(device)
    }

    if os.path.exists(exp_log_path):
        df_log = pd.read_csv(exp_log_path)
        df_log = pd.concat([df_log, pd.DataFrame([exp_entry])], ignore_index=True)
    else:
        df_log = pd.DataFrame([exp_entry])

    df_log.to_csv(exp_log_path, index=False)
    logger.info(f"Updated experiment log at {exp_log_path}")

if __name__ == "__main__":
    main()
