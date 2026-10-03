import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.visualization.plots import create_segmentation_overlay
from src.inference.predict import load_trained_model, predict_ct_volume
from src.utils.logger import get_logger

logger = get_logger("VerifyVisSpace")

def verify_visualization_space():
    output_dir = "results/raw_ct_verification"
    os.makedirs(output_dir, exist_ok=True)

    ct_path = "kits21/data/case_00000/imaging.nii.gz"
    seg_path = "kits21/data/case_00000/aggregated_MAJ_seg.nii.gz"
    ckpt_path = "checkpoints/segresnet/best_model.pth"

    logger.info("==================================================")
    logger.info("VISUALIZATION SPACE VERIFICATION FOR CASE_00000")
    logger.info("==================================================")

    # 1. Load Original CT NIfTI
    orig_nifti = nib.load(ct_path)
    ct_data = orig_nifti.get_fdata()
    spacing = tuple([float(x) for x in orig_nifti.header.get_zooms()[:3]])

    logger.info(f"Original CT Shape: {ct_data.shape}")
    logger.info(f"Original CT Dtype: {ct_data.dtype}")
    logger.info(f"Original CT Voxel Spacing (mm): {spacing}")
    logger.info(f"Minimum HU: {np.min(ct_data):.2f} | Maximum HU: {np.max(ct_data):.2f}")
    logger.info(f"HU Percentiles (1%, 50%, 99%): {np.percentile(ct_data, [1, 50, 99])}")

    # 2. Load GT and Run Inference
    gt_data = nib.load(seg_path).get_fdata() if os.path.exists(seg_path) else None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(ckpt_path, device=device)
    pred_data, _ = predict_ct_volume(model, ct_path, device=device)

    logger.info(f"Ground Truth Shape: {gt_data.shape if gt_data is not None else 'None'}")
    logger.info(f"Prediction Original Space Shape: {pred_data.shape}")

    # Assertion check
    assert ct_data.shape == pred_data.shape, f"CT shape {ct_data.shape} != Prediction shape {pred_data.shape}"
    logger.info("✓ Assertion Passed: ct_original.shape == prediction_original.shape")

    # 3. Save Raw CT Axial Slices
    D, H, W = ct_data.shape
    sample_slices = [D // 4, D // 2, (3 * D) // 4]
    
    for s_idx in sample_slices:
        ct_slice = ct_data[s_idx, :, :]
        plt.figure(figsize=(6, 6))
        # Windowed soft-tissue [-160, 240] HU
        ct_win = np.clip(ct_slice, -160, 240)
        plt.imshow(ct_win, cmap="gray", origin="upper")
        plt.title(f"Case 00000 Raw CT — Slice #{s_idx} ({H}x{W})")
        plt.axis("off")
        slice_out_path = os.path.join(output_dir, f"ct_slice_{s_idx}.png")
        plt.savefig(slice_out_path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"Saved raw CT slice image: {slice_out_path}")

    # 4. Save High-Resolution RGB Segmentation Overlay
    # Target slice with kidney/tumor voxels
    fg_counts = np.sum(pred_data > 0, axis=(1, 2))
    best_slice = int(np.argmax(fg_counts)) if np.max(fg_counts) > 0 else D // 2
    
    ct_slice_best = ct_data[best_slice, :, :]
    pred_slice_best = pred_data[best_slice, :, :]

    rgb_overlay = create_segmentation_overlay(
        ct_slice=ct_slice_best,
        mask_slice=pred_slice_best,
        alpha=0.45
    )

    overlay_out_path = os.path.join(output_dir, f"ct_overlay_slice_{best_slice}.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_overlay, origin="upper")
    plt.title(f"High-Res Original CT Overlay — Slice #{best_slice}\nKidney (Green) | Tumor (Red) | Cyst (Blue)")
    plt.axis("off")
    plt.savefig(overlay_out_path, bbox_inches="tight", dpi=200)
    plt.close()
    logger.info(f"Saved high-res overlay image: {overlay_out_path}")
    logger.info("Verification completed successfully!")

if __name__ == "__main__":
    verify_visualization_space()
