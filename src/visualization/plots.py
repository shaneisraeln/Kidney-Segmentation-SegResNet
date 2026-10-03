import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any
from skimage.measure import marching_cubes
from src.utils.logger import get_logger

logger = get_logger("Visualization")

CLASS_NAMES = {
    1: "Kidney Parenchyma",
    2: "Kidney Tumor",
    3: "Kidney Cyst"
}

CLASS_COLORS = {
    1: "#00FF64",  # Vibrant Green (0, 255, 100)
    2: "#FF2850",  # Vibrant Red (255, 40, 80)
    3: "#3296FF"   # Bright Blue (50, 150, 255)
}

CLASS_RGB_COLORS = {
    1: (0, 255, 100),   # Kidney Parenchyma - Vibrant Green
    2: (255, 40, 80),   # Renal Mass / Tumor - Vibrant Red
    3: (50, 150, 255)   # Renal Cyst - Bright Blue
}

def create_segmentation_overlay(
    ct_slice: np.ndarray,
    mask_slice: Optional[np.ndarray] = None,
    alpha: float = 0.45,
    show_kidney: bool = True,
    show_tumor: bool = True,
    show_cyst: bool = True,
    window_min: float = -150.0,
    window_max: float = 350.0
) -> np.ndarray:
    """
    Creates an explicit 3-channel RGB uint8 composite image (H, W, 3) where the grayscale
    CT background is windowed/normalized first, and multi-class categorical segmentations are
    alpha-blended on top using distinct RGB colors.

    Classes:
    - 1: Kidney (Green: RGB(0, 255, 100))
    - 2: Tumor (Red: RGB(255, 40, 80))
    - 3: Cyst (Blue: RGB(50, 150, 255))

    Returns:
        rgb_overlay: uint8 numpy array of shape (H, W, 3).
    """
    ct = ct_slice.astype(np.float32)
    # CT intensity windowing & normalization to 0 - 255 uint8
    if ct.max() > 1.0 or ct.min() < 0.0:
        ct = np.clip(ct, window_min, window_max)
        ct = (ct - window_min) / (window_max - window_min)
    else:
        ct = np.clip(ct, 0.0, 1.0)
    
    ct_gray = (ct * 255.0).astype(np.uint8)
    
    # 3-channel grayscale CT base image: (H, W, 3)
    rgb = np.stack([ct_gray, ct_gray, ct_gray], axis=-1).astype(np.float32)

    if mask_slice is None:
        return rgb.astype(np.uint8)

    mask = mask_slice.astype(np.int32)
    
    visibility = {
        1: show_kidney,
        2: show_tumor,
        3: show_cyst
    }

    # Opacity boosts for small lesions (e.g. 52-voxel tumor) so they pop out visually
    class_alphas = {
        1: alpha,
        2: min(1.0, alpha + 0.20),
        3: alpha
    }

    for c, color in CLASS_RGB_COLORS.items():
        if not visibility.get(c, True):
            continue
        
        class_mask = (mask == c)
        if not np.any(class_mask):
            continue
        
        a = class_alphas.get(c, alpha)
        color_arr = np.array(color, dtype=np.float32)
        
        # Explicit alpha blending: output = (1 - alpha) * CT_RGB + alpha * CLASS_COLOR
        rgb[class_mask] = (1.0 - a) * rgb[class_mask] + a * color_arr

    return np.clip(rgb, 0, 255).astype(np.uint8)

def create_error_map_overlay(
    ct_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    alpha: float = 0.50,
    window_min: float = -150.0,
    window_max: float = 350.0
) -> np.ndarray:
    """
    Creates an explicit 3-channel RGB uint8 composite image (H, W, 3) for voxel agreement/error mapping:
    - True Positive (TP): Green RGB(0, 255, 100)
    - False Positive (FP): Red RGB(255, 40, 80)
    - False Negative (FN): Orange RGB(255, 160, 0)
    """
    ct = ct_slice.astype(np.float32)
    if ct.max() > 1.0 or ct.min() < 0.0:
        ct = np.clip(ct, window_min, window_max)
        ct = (ct - window_min) / (window_max - window_min)
    else:
        ct = np.clip(ct, 0.0, 1.0)
    
    ct_gray = (ct * 255.0).astype(np.uint8)
    rgb = np.stack([ct_gray, ct_gray, ct_gray], axis=-1).astype(np.float32)

    gt_bin = (gt_slice > 0)
    pr_bin = (pred_slice > 0)

    tp_mask = gt_bin & pr_bin
    fp_mask = (~gt_bin) & pr_bin
    fn_mask = gt_bin & (~pr_bin)

    error_configs = [
        (tp_mask, (0, 255, 100), alpha),        # Green = TP
        (fp_mask, (255, 40, 80), alpha + 0.15), # Red = FP
        (fn_mask, (255, 160, 0), alpha + 0.15)  # Orange = FN
    ]

    for mask, color, a in error_configs:
        if np.any(mask):
            a_clamped = min(1.0, max(0.0, a))
            color_arr = np.array(color, dtype=np.float32)
            rgb[mask] = (1.0 - a_clamped) * rgb[mask] + a_clamped * color_arr

    return np.clip(rgb, 0, 255).astype(np.uint8)


def generate_3d_marching_cubes_mesh(
    seg_vol: np.ndarray,
    spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0),
    visible_classes: List[int] = [1, 2, 3],
    opacities: Optional[Dict[int, float]] = None
) -> Tuple[go.Figure, List[str]]:
    """
    Generates a Plotly 3D isosurface mesh scene using Marching Cubes surface extraction
    with physical spacing preservation and bounding-box camera framing.
    
    Returns:
        fig: Plotly 3D Figure object
        empty_info: List of human-readable messages for absent/empty classes
    """
    if opacities is None:
        opacities = {1: 0.35, 2: 0.85, 3: 0.85}

    fig = go.Figure()
    empty_info = []
    all_verts_list = []

    # Adaptive stride to prevent browser JSON string memory overflow (RangeError)
    max_dim = max(seg_vol.shape)
    if max_dim >= 400:
        stride = 3
    elif max_dim >= 200:
        stride = 2
    else:
        stride = 1

    effective_spacing = (spacing[0] * stride, spacing[1] * stride, spacing[2] * stride)

    for c in [1, 2, 3]:
        if c not in visible_classes:
            continue

        mask = (seg_vol == c)
        voxel_count = int(np.sum(mask))

        if voxel_count < 10:
            empty_info.append(f"**{CLASS_NAMES[c]}:** No voxels detected in this scan.")
            continue

        try:
            mask_strided = mask[::stride, ::stride, ::stride]
            if not np.any(mask_strided):
                mask_strided = mask
                use_spacing = spacing
            else:
                use_spacing = effective_spacing

            # Marching Cubes surface extraction with physical spacing
            verts, faces, _, _ = marching_cubes(
                mask_strided.astype(np.float32),
                level=0.5,
                spacing=use_spacing
            )
            # verts coordinates (Z_mm, Y_mm, X_mm)
            z_coords = verts[:, 0].astype(np.float32)
            y_coords = verts[:, 1].astype(np.float32)
            x_coords = verts[:, 2].astype(np.float32)
            all_verts_list.append((x_coords, y_coords, z_coords))

            color_hex = str(CLASS_COLORS.get(c, "#00FF64"))
            opacity = float(opacities.get(c, 0.5))

            mesh = go.Mesh3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color_hex,
                opacity=opacity,
                name=CLASS_NAMES[c],
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.3, specular=0.5),
                hovertemplate=f"<b>{CLASS_NAMES[c]}</b><br>X: %{{x:.1f}} mm<br>Y: %{{y:.1f}} mm<br>Z: %{{z:.1f}} mm<extra></extra>"
            )
            fig.add_trace(mesh)

        except Exception as e:
            logger.warning(f"Could not extract marching cubes mesh for class {c}: {e}")
            empty_info.append(f"**{CLASS_NAMES[c]}:** Surface extraction skipped ({e}).")

    if all_verts_list:
        # Calculate bounding box to frame camera cleanly
        all_x = np.concatenate([v[0] for v in all_verts_list])
        all_y = np.concatenate([v[1] for v in all_verts_list])
        all_z = np.concatenate([v[2] for v in all_verts_list])

        center_x = (np.min(all_x) + np.max(all_x)) / 2.0
        center_y = (np.min(all_y) + np.max(all_y)) / 2.0
        center_z = (np.min(all_z) + np.max(all_z)) / 2.0

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Sagittal (X - mm)", backgroundcolor="#0F172A", gridcolor="#334155"),
                yaxis=dict(title="Coronal (Y - mm)", backgroundcolor="#0F172A", gridcolor="#334155"),
                zaxis=dict(title="Axial (Z - mm)", backgroundcolor="#0F172A", gridcolor="#334155"),
                aspectmode="data",
                camera=dict(
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10),
            height=650,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor="rgba(255, 255, 255, 0.2)",
                borderwidth=1
            )
        )
    else:
        fig.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            template="plotly_dark",
            height=400,
            annotations=[dict(
                text="No non-background structures available for 3D rendering",
                showarrow=False,
                font=dict(size=16, color="#94A3B8")
            )]
        )

    return fig, empty_info

def plot_training_curves(
    train_losses: List[float],
    val_dices: List[float],
    output_path: str = "results/figures/training_curves.png"
):
    """
    Plots training loss and validation Dice score curves over epochs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, linewidth=2, label="Train Loss")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Dice', color=color)
    ax2.plot(epochs, val_dices, color=color, linewidth=2, linestyle="--", label="Val Mean Dice")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training Loss and Validation Dice Curves")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved training curves plot to {output_path}")

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: str = "results/figures/model_comparison.png"
):
    """
    Plots comparison bar charts across 2D Baseline, 3D U-Net, and SegResNet.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if "model" not in comparison_df.columns:
        logger.error("Comparison dataframe missing 'model' column")
        return

    metrics_cols = [c for c in ["mean_dice", "dice_kidney", "dice_tumor", "dice_cyst"] if c in comparison_df.columns]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df["model"]))
    width = 0.2

    for idx, col in enumerate(metrics_cols):
        ax.bar(x + idx * width, comparison_df[col], width, label=col.replace("_", " ").title())

    ax.set_ylabel("Dice Score")
    ax.set_title("Model Architecture Performance Comparison")
    ax.set_xticks(x + width * (len(metrics_cols) - 1) / 2)
    ax.set_xticklabels(comparison_df["model"])
    ax.legend()
    ax.set_ylim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved model comparison plot to {output_path}")

def plot_triplane_views(
    ct_volume: np.ndarray,
    seg_mask: Optional[np.ndarray] = None,
    output_path: str = "results/figures/triplane_view.png"
):
    """
    Plots Axial, Coronal, and Sagittal cross-sectional views.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    D, H, W = ct_volume.shape
    
    axial_idx = D // 2
    coronal_idx = H // 2
    sagittal_idx = W // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial
    axes[0].imshow(ct_volume[axial_idx, :, :], cmap="gray", origin="lower")
    if seg_mask is not None:
        axes[0].imshow(seg_mask[axial_idx, :, :], cmap="jet", alpha=0.4, origin="lower")
    axes[0].set_title(f"Axial Plane (Z={axial_idx})")
    axes[0].axis("off")

    # Coronal
    axes[1].imshow(ct_volume[:, coronal_idx, :], cmap="gray", origin="lower")
    if seg_mask is not None:
        axes[1].imshow(seg_mask[:, coronal_idx, :], cmap="jet", alpha=0.4, origin="lower")
    axes[1].set_title(f"Coronal Plane (Y={coronal_idx})")
    axes[1].axis("off")

    # Sagittal
    axes[2].imshow(ct_volume[:, :, sagittal_idx], cmap="gray", origin="lower")
    if seg_mask is not None:
        axes[2].imshow(seg_mask[:, :, sagittal_idx], cmap="jet", alpha=0.4, origin="lower")
    axes[2].set_title(f"Sagittal Plane (X={sagittal_idx})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved triplane visualization to {output_path}")
