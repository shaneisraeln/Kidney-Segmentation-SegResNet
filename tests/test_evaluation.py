import pytest
import torch
import numpy as np
from src.training.metrics import SegmentationMetrics
from src.training.losses import build_loss, WeightedDiceCELoss
from src.visualization.plots import create_segmentation_overlay, create_error_map_overlay

def test_loss_builder():
    dice_ce_loss = build_loss("dice_ce", num_classes=4, class_weights=[0.2, 1.0, 3.0, 3.0])
    pred = torch.randn(2, 4, 32, 32, 32)
    target = torch.randint(0, 4, (2, 1, 32, 32, 32))
    total_loss, d_loss, ce_loss = dice_ce_loss(pred, target)
    assert not torch.isnan(total_loss), "Total loss value is NaN"
    assert total_loss >= 0.0, "Total loss should be non-negative"
    assert d_loss >= 0.0
    assert ce_loss >= 0.0

def test_metrics_and_hierarchical_calculation():
    metrics = SegmentationMetrics(num_classes=4)
    pred = torch.zeros(1, 4, 16, 16, 16)
    pred[:, 1, :, :, :] = 10.0 # High confidence for Kidney (class 1)
    
    target = torch.ones(1, 1, 16, 16, 16, dtype=torch.int64) # Class 1 ground truth
    
    metrics.update(pred, target)
    res = metrics.compute()
    assert res["dice_kidney"] > 0.95, f"Expected Kidney Dice > 0.95, got {res['dice_kidney']}"
    assert "kits_composite_kidney_dice" in res
    assert res["kits_composite_kidney_dice"] > 0.95

def test_segmentation_overlay_is_rgb():
    """
    Unit test verifying that create_segmentation_overlay returns a 3-channel uint8 RGB image
    and that mask regions contain explicit color channels (non-grayscale), while background pixels remain grayscale.
    """
    H, W = 64, 64
    ct_slice = np.full((H, W), 100.0, dtype=np.float32) # Homogeneous CT slice (HU=100)
    mask_slice = np.zeros((H, W), dtype=np.int32)
    
    # Class 1 (Kidney) region
    mask_slice[10:20, 10:20] = 1
    # Class 2 (Tumor) region
    mask_slice[30:40, 30:40] = 2
    # Class 3 (Cyst) region
    mask_slice[50:60, 50:60] = 3

    rgb_overlay = create_segmentation_overlay(ct_slice, mask_slice, alpha=0.5)

    assert rgb_overlay.ndim == 3, f"Expected 3D array (H, W, 3), got ndim {rgb_overlay.ndim}"
    assert rgb_overlay.shape == (H, W, 3), f"Expected shape ({H}, {W}, 3), got {rgb_overlay.shape}"
    assert rgb_overlay.dtype == np.uint8, f"Expected dtype uint8, got {rgb_overlay.dtype}"

    # Verify background pixels remain grayscale (R == G == B)
    bg_pixel = rgb_overlay[0, 0]
    assert bg_pixel[0] == bg_pixel[1] == bg_pixel[2], f"Background pixel should be grayscale, got {bg_pixel}"

    # Verify Class 1 (Kidney - Green) is non-grayscale with Green dominant
    k_pixel = rgb_overlay[15, 15]
    assert not (k_pixel[0] == k_pixel[1] == k_pixel[2]), f"Class 1 pixel must be colored, got {k_pixel}"
    assert k_pixel[1] > k_pixel[0] and k_pixel[1] > k_pixel[2], f"Class 1 (Kidney) pixel should be green-dominated, got {k_pixel}"

    # Verify Class 2 (Tumor - Red) is non-grayscale with Red dominant
    t_pixel = rgb_overlay[35, 35]
    assert not (t_pixel[0] == t_pixel[1] == t_pixel[2]), f"Class 2 pixel must be colored, got {t_pixel}"
    assert t_pixel[0] > t_pixel[1] and t_pixel[0] > t_pixel[2], f"Class 2 (Tumor) pixel should be red-dominated, got {t_pixel}"

    # Verify Class 3 (Cyst - Blue) is non-grayscale with Blue dominant
    c_pixel = rgb_overlay[55, 55]
    assert not (c_pixel[0] == c_pixel[1] == c_pixel[2]), f"Class 3 pixel must be colored, got {c_pixel}"
    assert c_pixel[2] > c_pixel[0] and c_pixel[2] > c_pixel[1], f"Class 3 (Cyst) pixel should be blue-dominated, got {c_pixel}"

if __name__ == "__main__":
    pytest.main([__file__])
