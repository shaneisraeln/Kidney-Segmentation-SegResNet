import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss
from typing import Optional, List, Tuple

class WeightedDiceCELoss(nn.Module):
    """
    Combined Class-Weighted Dice and Cross-Entropy Loss for multi-class 3D CT segmentation.
    Addresses extreme foreground/background class imbalance (Kidney, Tumor, Cyst).
    """
    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[List[float]] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        include_background: bool = False
    ):
        super(WeightedDiceCELoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = torch.tensor([0.2, 1.0, 3.0, 3.0], dtype=torch.float32)
            
        self.dice_loss = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=include_background
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            y_pred: Logits tensor (B, C, D, H, W)
            y_true: Target tensor (B, 1, D, H, W) with integer class indices
        Returns:
            total_loss, dice_loss_val, ce_loss_val
        """
        device = y_pred.device
        weights = self.class_weights.to(device)
        
        # 1. Compute Dice Loss
        d_loss = self.dice_loss(y_pred, y_true)
        
        # 2. Compute Class-Weighted Cross Entropy Loss
        target_flat = y_true.squeeze(1).long() # (B, D, H, W)
        ce_loss = F.cross_entropy(y_pred, target_flat, weight=weights)
        
        total_loss = self.lambda_dice * d_loss + self.lambda_ce * ce_loss
        return total_loss, d_loss, ce_loss

def build_loss(
    loss_name: str = "dice_ce",
    num_classes: int = 4,
    class_weights: Optional[List[float]] = None
) -> nn.Module:
    """
    Build loss function for multi-class segmentation.
    """
    name = loss_name.lower()
    if name in ["dice_ce", "dicece", "weighted_dice_ce"]:
        return WeightedDiceCELoss(num_classes=num_classes, class_weights=class_weights)
    elif name == "dice":
        return DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    else:
        return WeightedDiceCELoss(num_classes=num_classes, class_weights=class_weights)
