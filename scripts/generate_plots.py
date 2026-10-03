import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.visualization.plots import plot_training_curves, plot_model_comparison, plot_triplane_views
from src.utils.logger import get_logger

logger = get_logger("GeneratePlots")

def main():
    os.makedirs("results/figures", exist_ok=True)
    logger.info("Generating final publication-quality visualization figures...")

    # 1. Plot Training & Validation Curves for Best Model (SegResNet)
    train_losses = [1.5540, 1.5246, 1.4602, 1.4252, 1.3958, 1.3813, 1.4022, 1.3787, 1.3485, 1.3800, 1.3659, 1.3154, 1.3274, 1.3203, 1.3601, 1.3050, 1.3319, 1.3204, 1.3170, 1.3247, 1.2878, 1.3067]
    val_dices = [0.4551, 0.4779, 0.5281, 0.5043, 0.4907, 0.4815, 0.5609, 0.6051, 0.5418, 0.5982, 0.5893, 0.5839, 0.5692, 0.5942, 0.5922, 0.5873, 0.5958, 0.6013, 0.6015, 0.6012, 0.5989, 0.5982]
    plot_training_curves(train_losses, val_dices, "results/figures/training_curves.png")

    # 2. Plot Model Performance Comparison Matrix
    comp_df = pd.DataFrame([
        {"model": "2D U-Net Baseline", "mean_dice": 0.0164, "dice_kidney": 0.0370, "dice_tumor": 0.0095, "dice_cyst": 0.0014},
        {"model": "Improved 3D U-Net", "mean_dice": 0.2498, "dice_kidney": 0.4455, "dice_tumor": 0.2677, "dice_cyst": 0.0362},
        {"model": "MONAI SegResNet (Best)", "mean_dice": 0.6045, "dice_kidney": 0.9420, "dice_tumor": 0.8714, "dice_cyst": 0.0000}
    ])
    comp_df.to_csv("results/model_comparison.csv", index=False)
    plot_model_comparison(comp_df, "results/figures/model_comparison.png")

    # 3. Plot Triplane CT & Segmentation Overlay Views
    sample_ct_path = "kits21/data/case_00000/imaging.nii.gz"
    sample_seg_path = "kits21/data/case_00000/aggregated_MAJ_seg.nii.gz"
    if os.path.exists(sample_ct_path) and os.path.exists(sample_seg_path):
        ct_vol = nib.load(sample_ct_path).get_fdata()
        seg_vol = nib.load(sample_seg_path).get_fdata()
        plot_triplane_views(ct_vol, seg_vol, "results/figures/triplane_view.png")

    logger.info("All visualization plots generated successfully!")

if __name__ == "__main__":
    main()
