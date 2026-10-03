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

logger = get_logger("AlignmentVerify")

def verify_alignment(case_id: str = "case_00000"):
    output_dir = "results/alignment_verification"
    os.makedirs(output_dir, exist_ok=True)

    ct_path = f"kits21/data/{case_id}/imaging.nii.gz"
    gt_path = f"kits21/data/{case_id}/aggregated_MAJ_seg.nii.gz"
    ckpt_path = "checkpoints/segresnet/best_model.pth"

    logger.info("==================================================")
    logger.info(f"SPATIAL ALIGNMENT VERIFICATION FOR {case_id.upper()}")
    logger.info("==================================================")

    # 1. Load Original CT
    nifti_ct = nib.load(ct_path)
    ct_data = nifti_ct.get_fdata()
    spacing = tuple([float(x) for x in nifti_ct.header.get_zooms()[:3]])
    axcodes = nib.orientations.aff2axcodes(nifti_ct.affine)

    logger.info(f"Original CT:")
    logger.info(f"  Shape: {ct_data.shape}")
    logger.info(f"  Voxel Spacing (mm): {spacing}")
    logger.info(f"  Orientation codes: {axcodes}")
    logger.info(f"  HU range: [{np.min(ct_data):.1f}, {np.max(ct_data):.1f}]")

    # 2. Load Ground Truth
    nifti_gt = nib.load(gt_path)
    gt_data = nifti_gt.get_fdata().astype(np.uint8)
    gt_unique, gt_counts = np.unique(gt_data, return_counts=True)

    logger.info(f"Ground Truth:")
    logger.info(f"  Shape: {gt_data.shape}")
    logger.info(f"  Unique labels & counts: {dict(zip(gt_unique, gt_counts))}")

    # 3. Model Inference & Restoration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(ckpt_path, device=device)
    
    pred_data, _ = predict_ct_volume(model, ct_path, device=device)
    pred_unique, pred_counts = np.unique(pred_data, return_counts=True)

    logger.info(f"Restored Prediction Original Space:")
    logger.info(f"  Shape: {pred_data.shape}")
    logger.info(f"  Unique labels & counts: {dict(zip(pred_unique, pred_counts))}")

    # 4. Assert Alignment
    assert ct_data.shape == gt_data.shape == pred_data.shape, "Shapes do not match!"
    logger.info("✓ Assertion Passed: ct_shape == gt_shape == pred_shape")

    # Select best axial slice with maximum GT/Prediction foreground voxels
    fg_gt = np.sum(gt_data > 0, axis=(1, 2))
    best_slice = int(np.argmax(fg_gt)) if np.max(fg_gt) > 0 else ct_data.shape[0] // 2

    logger.info(f"Selected Diagnostic Slice: #{best_slice}")
    
    ct_slice = ct_data[best_slice, :, :]
    gt_slice = gt_data[best_slice, :, :]
    pr_slice = pred_data[best_slice, :, :]

    logger.info(f"Slice #{best_slice} GT counts: {dict(zip(*np.unique(gt_slice, return_counts=True)))}")
    logger.info(f"Slice #{best_slice} Pred counts: {dict(zip(*np.unique(pr_slice, return_counts=True)))}")

    # Image 1: Raw CT Only (Windowed [-160, 240] HU)
    ct_win = np.clip(ct_slice, -160, 240)
    plt.figure(figsize=(7, 7))
    plt.imshow(ct_win, cmap="gray", origin="upper")
    plt.title(f"01 Raw CT Slice #{best_slice}")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "01_raw_ct.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # Image 2: Ground Truth Overlay
    rgb_gt = create_segmentation_overlay(ct_slice, gt_slice, alpha=0.45)
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb_gt, origin="upper")
    plt.title(f"02 Ground Truth Overlay Slice #{best_slice}")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "02_ground_truth.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # Image 3: Model Prediction Overlay
    rgb_pr = create_segmentation_overlay(ct_slice, pr_slice, alpha=0.45)
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb_pr, origin="upper")
    plt.title(f"03 Model Prediction Overlay Slice #{best_slice}")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "03_prediction.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # Image 4: Side-by-Side (CT | Ground Truth | Prediction)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(ct_win, cmap="gray", origin="upper")
    axes[0].set_title("Original CT (HU -160 to 240)")
    axes[0].axis("off")

    axes[1].imshow(rgb_gt, origin="upper")
    axes[1].set_title("Ground Truth Annotation")
    axes[1].axis("off")

    axes[2].imshow(rgb_pr, origin="upper")
    axes[2].set_title("Model Prediction (SegResNet)")
    axes[2].axis("off")

    plt.suptitle(f"Spatial Alignment Verification — {case_id} Slice #{best_slice}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_side_by_side.png"), bbox_inches="tight", dpi=200)
    plt.close()

    # Image 5: Mask Only (Categorical Prediction)
    plt.figure(figsize=(7, 7))
    plt.imshow(pr_slice, cmap="tab10", vmin=0, vmax=3, origin="upper")
    plt.title(f"05 Discrete Categorical Mask Slice #{best_slice}")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "05_mask_only.png"), bbox_inches="tight", dpi=150)
    plt.close()

    logger.info("Successfully generated all 5 verification images in results/alignment_verification/")

if __name__ == "__main__":
    verify_alignment("case_00000")
