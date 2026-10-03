import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

class SegmentationMetrics:
    """
    Computes per-class and hierarchical KiTS metrics (Dice, IoU, HD95).
    Hierarchical KiTS regions:
    - Composite Kidney (Kidney + Tumor + Cyst): (label >= 1)
    - Kidney Mass (Tumor + Cyst): (label == 2) | (label == 3)
    - Tumor Only: (label == 2)
    """
    def __init__(self, num_classes: int = 4, include_background: bool = False):
        self.num_classes = num_classes
        self.include_background = include_background
        
        self.dice_metric = DiceMetric(include_background=include_background, reduction="none")
        self.iou_metric = MeanIoU(include_background=include_background, reduction="none")
        self.hd95_metric = HausdorffDistanceMetric(include_background=include_background, percentile=95, reduction="none")
        self.post_label = AsDiscrete(to_onehot=num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

        # Hierarchical KiTS metrics
        self.hier_dice = DiceMetric(include_background=False, reduction="none")

    def reset(self):
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.hd95_metric.reset()
        self.hier_dice.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        y_pred: (B, C, D, H, W) logits or probabilities
        y_true: (B, 1, D, H, W) target class indices
        """
        y_pred_argmax = torch.argmax(y_pred, dim=1, keepdim=True) # (B, 1, D, H, W)
        
        y_pred_onehot = [self.post_pred(p) for p in y_pred]
        y_true_onehot = [self.post_label(t) for t in y_true]
        
        y_pred_tensor = torch.stack(y_pred_onehot)
        y_true_tensor = torch.stack(y_true_onehot)
        
        self.dice_metric(y_pred_tensor, y_true_tensor)
        self.iou_metric(y_pred_tensor, y_true_tensor)
        
        try:
            self.hd95_metric(y_pred_tensor, y_true_tensor)
        except Exception:
            pass

        # Calculate Hierarchical Regions
        # Region 0: Composite Kidney (Kidney + Mass) -> label >= 1
        # Region 1: Mass (Tumor + Cyst) -> label == 2 or 3
        # Region 2: Tumor Only -> label == 2
        pred_composite = (y_pred_argmax >= 1).float()
        true_composite = (y_true >= 1).float()
        
        pred_mass = ((y_pred_argmax == 2) | (y_pred_argmax == 3)).float()
        true_mass = ((y_true == 2) | (y_true == 3)).float()
        
        pred_tumor = (y_pred_argmax == 2).float()
        true_tumor = (y_true == 2).float()
        
        pred_hier = torch.cat([pred_composite, pred_mass, pred_tumor], dim=1) # (B, 3, D, H, W)
        true_hier = torch.cat([true_composite, true_mass, true_tumor], dim=1)
        
        # Convert to 2-channel binary onehot per region for MONAI DiceMetric
        for i in range(3):
            p_bin = torch.cat([1 - pred_hier[:, i:i+1], pred_hier[:, i:i+1]], dim=1)
            t_bin = torch.cat([1 - true_hier[:, i:i+1], true_hier[:, i:i+1]], dim=1)
            self.hier_dice(p_bin, t_bin)

    def compute(self, class_names: List[str] = ["Kidney", "Tumor", "Cyst"]) -> Dict[str, Any]:
        """
        Computes metric statistics across per-class and hierarchical KiTS regions.
        """
        dice_vals = self.dice_metric.aggregate()
        iou_vals = self.iou_metric.aggregate()
        
        results = {}
        
        dice_np = dice_vals.cpu().numpy() if isinstance(dice_vals, torch.Tensor) else np.array(dice_vals)
        iou_np = iou_vals.cpu().numpy() if isinstance(iou_vals, torch.Tensor) else np.array(iou_vals)
        
        class_dice = np.nanmean(dice_np, axis=0) if dice_np.ndim > 1 else dice_np
        class_iou = np.nanmean(iou_np, axis=0) if iou_np.ndim > 1 else iou_np
        
        for idx, name in enumerate(class_names):
            if idx < len(class_dice):
                results[f"dice_{name.lower()}"] = float(class_dice[idx])
                results[f"iou_{name.lower()}"] = float(class_iou[idx])
                
        results["mean_dice"] = float(np.nanmean(class_dice))
        results["mean_iou"] = float(np.nanmean(class_iou))
        
        try:
            hd95_vals = self.hd95_metric.aggregate()
            hd95_np = hd95_vals.cpu().numpy() if isinstance(hd95_vals, torch.Tensor) else np.array(hd95_vals)
            class_hd95 = np.nanmean(hd95_np, axis=0) if hd95_np.ndim > 1 else hd95_np
            for idx, name in enumerate(class_names):
                if idx < len(class_hd95):
                    results[f"hd95_{name.lower()}"] = float(class_hd95[idx])
            results["mean_hd95"] = float(np.nanmean(class_hd95))
        except Exception:
            results["mean_hd95"] = float('nan')

        # Hierarchical KiTS region scores
        try:
            hier_vals = self.hier_dice.aggregate()
            hier_np = hier_vals.cpu().numpy() if isinstance(hier_vals, torch.Tensor) else np.array(hier_vals)
            hier_means = np.nanmean(hier_np, axis=0) if hier_np.ndim > 1 else hier_np
            
            results["kits_composite_kidney_dice"] = float(hier_means[0]) if len(hier_means) > 0 else float('nan')
            results["kits_mass_dice"] = float(hier_means[1]) if len(hier_means) > 1 else float('nan')
            results["kits_tumor_dice"] = float(hier_means[2]) if len(hier_means) > 2 else float('nan')
        except Exception:
            pass

        return results
