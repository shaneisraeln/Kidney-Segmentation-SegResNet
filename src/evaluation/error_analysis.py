import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("ErrorAnalysis")

def run_error_analysis(
    metrics_csv_path: str = "results/test_metrics.csv",
    data_dir: str = "kits21/data",
    output_dir: str = "results/error_cases",
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Identifies best and worst performing cases and generates visual overlay error reports.
    """
    if not os.path.exists(metrics_csv_path):
        logger.error(f"Metrics CSV not found at {metrics_csv_path}. Run test evaluation first.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metrics_csv_path)

    if "mean_dice" not in df.columns:
        logger.error("mean_dice column missing from test metrics CSV.")
        return {}

    # Sort by mean_dice
    df_sorted = df.sort_values(by="mean_dice", ascending=False)
    best_cases = df_sorted.head(top_k)
    worst_cases = df_sorted.tail(top_k)

    logger.info(f"Top {top_k} Best Cases:\n{best_cases[['case_id', 'mean_dice']]}")
    logger.info(f"Top {top_k} Worst Cases:\n{worst_cases[['case_id', 'mean_dice']]}")

    # Generate figure overlays for best & worst cases
    def generate_case_figure(case_row, tag: str):
        case_id = case_row["case_id"]
        case_dir = os.path.join(data_dir, case_id)
        img_path = os.path.join(case_dir, "imaging.nii.gz")
        seg_path = os.path.join(case_dir, "aggregated_MAJ_seg.nii.gz")

        if not (os.path.exists(img_path) and os.path.exists(seg_path)):
            return

        img_data = nib.load(img_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()

        # Find axial slice with largest kidney/tumor ROI
        slice_sums = np.sum(seg_data > 0, axis=(0, 1))
        best_slice_idx = int(np.argmax(slice_sums))

        ct_slice = img_data[:, :, best_slice_idx]
        gt_slice = seg_data[:, :, best_slice_idx]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ct_slice.T, cmap="gray", origin="lower")
        axes[0].set_title(f"CT Slice (Z={best_slice_idx})")
        axes[0].axis("off")

        axes[1].imshow(ct_slice.T, cmap="gray", origin="lower")
        axes[1].imshow(gt_slice.T, cmap="jet", alpha=0.5, origin="lower")
        axes[1].set_title(f"Ground Truth Overlay")
        axes[1].axis("off")

        axes[2].imshow(gt_slice.T > 0, cmap="inferno", origin="lower")
        axes[2].set_title(f"ROI Mask (Dice: {case_row.get('mean_dice', 0):.3f})")
        axes[2].axis("off")

        plt.suptitle(f"{tag.upper()} CASE: {case_id}", fontsize=14)
        plt.tight_layout()
        out_fig_path = os.path.join(output_dir, f"{tag}_{case_id}.png")
        plt.savefig(out_fig_path, dpi=150)
        plt.close()
        logger.info(f"Saved error analysis visualization to {out_fig_path}")

    for idx, row in best_cases.iterrows():
        generate_case_figure(row, "best")

    for idx, row in worst_cases.iterrows():
        generate_case_figure(row, "worst")

    return {
        "best_cases": best_cases["case_id"].tolist(),
        "worst_cases": worst_cases["case_id"].tolist()
    }

if __name__ == "__main__":
    run_error_analysis()
