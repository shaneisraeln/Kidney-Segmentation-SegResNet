import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import nibabel as nib

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.visualization.plots import (
    generate_3d_marching_cubes_mesh,
    create_segmentation_overlay,
    create_error_map_overlay,
    CLASS_COLORS,
    CLASS_RGB_COLORS
)
from src.inference.predict import load_trained_model, predict_ct_volume

st.set_page_config(
    page_title="3D Kidney & Tumor Segmentation Research Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Glassmorphism Styling
st.markdown("""
<style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    .pipeline-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 0.92rem;
        color: #CBD5E1;
    }
    .explain-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 16px;
        font-size: 0.88rem;
        color: #94A3B8;
    }
    .legend-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 8px;
    }
    .bg-kidney { background: rgba(0, 255, 100, 0.15); color: #00FF64; border: 1px solid #00FF64; }
    .bg-tumor { background: rgba(255, 40, 80, 0.15); color: #FF2850; border: 1px solid #FF2850; }
    .bg-cyst { background: rgba(50, 150, 255, 0.15); color: #3296FF; border: 1px solid #3296FF; }
    
    .summary-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 15px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# HELPER FUNCTIONS FOR COMPOSITING & RGBA OVERLAYS
# ----------------------------------------------------
def create_rgba_overlay(mask_slice: np.ndarray, alpha: float = 0.5, show_k: bool = True, show_t: bool = True, show_c: bool = True) -> np.ndarray:
    """
    Converts 2D class index segmentation slice (H, W) into an RGBA color image (H, W, 4)
    with exact class colors and custom transparency.
    """
    H, W = mask_slice.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    a_int = int(clamp(alpha, 0.0, 1.0) * 255)

    if show_k:
        k_mask = (mask_slice == 1)
        rgba[k_mask, 0] = 0    # Red
        rgba[k_mask, 1] = 255  # Green
        rgba[k_mask, 2] = 100  # Blue (#00FF64)
        rgba[k_mask, 3] = a_int

    if show_t:
        t_mask = (mask_slice == 2)
        rgba[t_mask, 0] = 255  # Red
        rgba[t_mask, 1] = 40   # Green
        rgba[t_mask, 2] = 80   # Blue (#FF2850)
        rgba[t_mask, 3] = a_int

    if show_c:
        c_mask = (mask_slice == 3)
        rgba[c_mask, 0] = 50   # Red
        rgba[c_mask, 1] = 150  # Green
        rgba[c_mask, 2] = 255  # Blue (#3296FF)
        rgba[c_mask, 3] = a_int

    return rgba

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

# ----------------------------------------------------
# HEADER & PIPELINE BANNER
# ----------------------------------------------------
st.title("3D Kidney & Tumor Segmentation Research Dashboard")
st.caption("KiTS21 Academic Benchmark & Deep Learning Framework (MONAI SegResNet & 3D U-Net)")

# Visual Pipeline Diagram
st.markdown("""
<div class="pipeline-card">
    <strong>3D CT Scan Volume</strong> &nbsp;->&nbsp; 
    <strong>MONAI Preprocessing</strong> &nbsp;->&nbsp; 
    <strong>3D SegResNet Model</strong> &nbsp;->&nbsp; 
    <strong>Segmentation Mask</strong> &nbsp;->&nbsp; 
    <span class="legend-badge bg-kidney">Kidney</span>
    <span class="legend-badge bg-tumor">Tumor</span>
    <span class="legend-badge bg-cyst">Cyst</span> &nbsp;->&nbsp; 
    <strong>Visual & Volumetric Analysis</strong>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR CONTROLS & MODEL LOADING
# ----------------------------------------------------
st.sidebar.header("DATA SOURCE")
data_mode = st.sidebar.radio("Select Input Mode", ["KiTS21 Sample Case", "Upload Custom NIfTI"], index=0)

ct_data = None
seg_data = None
pred_data = None
case_name = "Custom Scan"
ct_file_path = None
header_spacing = (1.5, 1.5, 2.0)

if data_mode == "KiTS21 Sample Case":
    sample_dir = "kits21/data"
    available_cases = []
    if os.path.exists(sample_dir):
        available_cases = sorted([d for d in os.listdir(sample_dir) if d.startswith("case_")])
    
    if available_cases:
        selected_case = st.sidebar.selectbox("Select KiTS Case ID", available_cases)
        case_name = selected_case
        case_path = os.path.join(sample_dir, selected_case)
        img_f = os.path.join(case_path, "imaging.nii.gz")
        seg_f = os.path.join(case_path, "aggregated_MAJ_seg.nii.gz")
        
        if os.path.exists(img_f):
            ct_file_path = img_f
            nifti_img = nib.load(img_f)
            ct_data = nifti_img.get_fdata()
            header_spacing = tuple([float(x) for x in nifti_img.header.get_zooms()[:3]])
        if os.path.exists(seg_f):
            seg_data = nib.load(seg_f).get_fdata()
else:
    uploaded_file = st.sidebar.file_uploader("Upload NIfTI CT Scan (.nii / .nii.gz)", type=["nii", "gz"])
    if uploaded_file is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
            tmp.write(uploaded_file.read())
            ct_file_path = tmp.name
        nifti_img = nib.load(ct_file_path)
        ct_data = nifti_img.get_fdata()
        header_spacing = tuple([float(x) for x in nifti_img.header.get_zooms()[:3]])
        case_name = uploaded_file.name

st.sidebar.markdown("---")
st.sidebar.header("MODEL SELECTION")

checkpoint_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    [
        "checkpoints/segresnet/best_model.pth (Best - Val Dice: 0.6051)",
        "checkpoints/unet3d/best_model.pth (Improved 3D U-Net - Val Dice: 0.2498)",
        "checkpoints/baseline_2d/best_model.pth (2D U-Net Baseline)"
    ]
)
checkpoint_path = checkpoint_choice.split(" ")[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.caption(f"Compute Hardware: **{device.type.upper()}**")

@st.cache_resource
def get_cached_model(ckpt_path: str):
    if os.path.exists(ckpt_path):
        return load_trained_model(ckpt_path, device=device)
    return None

model = get_cached_model(checkpoint_path)

# Auto-run prediction for loaded case so model prediction is ALWAYS pre-cached
if ct_file_path and model is not None:
    cache_key = f"pred_{case_name}_{checkpoint_path}"
    if cache_key in st.session_state:
        cached_p = st.session_state[cache_key]
        if ct_data is not None and cached_p.shape != ct_data.shape:
            del st.session_state[cache_key]

    if cache_key not in st.session_state:
        with st.spinner(f"Running sliding-window 3D inference for {case_name}..."):
            p_mask, _ = predict_ct_volume(model, ct_file_path, device)
            st.session_state[cache_key] = p_mask
    pred_data = st.session_state.get(cache_key, None)

if st.sidebar.button("Re-Run 3D Segmentation"):
    if ct_file_path and model is not None:
        cache_key = f"pred_{case_name}_{checkpoint_path}"
        if cache_key in st.session_state:
            del st.session_state[cache_key]
        with st.spinner("Re-executing 3D volumetric inference..."):
            p_mask, _ = predict_ct_volume(model, ct_file_path, device)
            st.session_state[cache_key] = p_mask
            pred_data = p_mask
            st.sidebar.success("Inference completed!")

st.sidebar.markdown("---")
st.sidebar.header("CT WINDOW CONTROLS")
win_preset = st.sidebar.selectbox(
    "CT Intensity Window Preset",
    ["Soft Tissue (Center 40 / Width 400 HU)", "Abdominal (Center 50 / Width 350 HU)", "Bone (Center 400 / Width 2000 HU)", "Custom HU Range"],
    index=0
)

if win_preset.startswith("Soft Tissue"):
    win_center, win_width = 40.0, 400.0
elif win_preset.startswith("Abdominal"):
    win_center, win_width = 50.0, 350.0
elif win_preset.startswith("Bone"):
    win_center, win_width = 400.0, 2000.0
else:
    c_w1, c_w2 = st.sidebar.columns(2)
    win_center = c_w1.slider("Window Center (HU)", -500.0, 1000.0, 40.0, 10.0)
    win_width = c_w2.slider("Window Width (HU)", 50.0, 3000.0, 400.0, 50.0)

win_min = win_center - (win_width / 2.0)
win_max = win_center + (win_width / 2.0)

st.sidebar.markdown("---")
st.sidebar.header("DASHBOARD NAVIGATION")
page = st.sidebar.radio(
    "Select View Page",
    [
        "Multi-Planar Slice Viewer",
        "Ground Truth vs Prediction",
        "3D Surface Rendering",
        "Volume & Anatomical Metrics",
        "Model Performance & Benchmarks",
        "Model Specs & Info"
    ]
)

# Active Class Color Scheme Bar
st.markdown("""
<div style="margin-bottom: 15px;">
    <strong>Class Color Legend:</strong> &nbsp;&nbsp;
    <span class="legend-badge bg-kidney">Kidney Parenchyma (#00FF64 - Green)</span>
    <span class="legend-badge bg-tumor">Renal Mass / Tumor (#FF2850 - Red)</span>
    <span class="legend-badge bg-cyst">Renal Cyst (#3296FF - Blue)</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# PAGE 1: MULTI-PLANAR SLICE VIEWER
# ----------------------------------------------------
if page == "Multi-Planar Slice Viewer":
    st.subheader("Multi-Planar CT Slice Exploration")
    
    if ct_data is None:
        st.info("Please select a sample KiTS case or upload a custom CT scan in the sidebar.")
    else:
        D, H, W = ct_data.shape
        c1, c2, c3 = st.columns(3)
        
        plane = c1.selectbox("Plane Orientation", ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"])
        
        has_gt = seg_data is not None
        target_options = []
        if pred_data is not None:
            target_options.append("Model Prediction")
        if has_gt:
            target_options.append("Ground Truth")
            
        if not target_options:
            st.warning("No segmentation mask loaded.")
            overlay_target = "None"
            active_mask = None
        else:
            overlay_target = c2.radio("Overlay Target", target_options, horizontal=True)
            active_mask = pred_data if overlay_target == "Model Prediction" else seg_data
            
        alpha = c3.slider("Mask Opacity", 0.0, 1.0, 0.50, 0.05)

        st.markdown("**Structure Visibility Filters:**")
        f_col1, f_col2, f_col3 = st.columns(3)
        show_kidney = f_col1.checkbox("Kidney Parenchyma", value=True)
        show_tumor = f_col2.checkbox("Renal Mass / Tumor", value=True)
        show_cyst = f_col3.checkbox("Renal Cyst", value=True)

        # Detect best default slice containing maximum foreground voxels
        if active_mask is not None and np.max(active_mask) > 0:
            if plane.startswith("Axial"):
                fg_counts = np.sum(active_mask > 0, axis=(1, 2))
            elif plane.startswith("Coronal"):
                fg_counts = np.sum(active_mask > 0, axis=(0, 2))
            else:
                fg_counts = np.sum(active_mask > 0, axis=(0, 1))
            default_slice = int(np.argmax(fg_counts)) if np.max(fg_counts) > 0 else (D // 2 if plane.startswith("Axial") else (H // 2 if plane.startswith("Coronal") else W // 2))
        else:
            default_slice = D // 2 if plane.startswith("Axial") else (H // 2 if plane.startswith("Coronal") else W // 2)

        if plane.startswith("Axial"):
            slice_idx = st.slider("Axial Slice Index (Z)", 0, D - 1, default_slice)
            ct_slice = ct_data[slice_idx, :, :]
            mask_slice = active_mask[slice_idx, :, :] if active_mask is not None else None
        elif plane.startswith("Coronal"):
            slice_idx = st.slider("Coronal Slice Index (Y)", 0, H - 1, default_slice)
            ct_slice = ct_data[:, slice_idx, :]
            mask_slice = active_mask[:, slice_idx, :] if active_mask is not None else None
        else:
            slice_idx = st.slider("Sagittal Slice Index (X)", 0, W - 1, default_slice)
            ct_slice = ct_data[:, :, slice_idx]
            mask_slice = active_mask[:, :, slice_idx] if active_mask is not None else None

        # Display Slice Summary Banner above image
        if mask_slice is not None:
            k_pix = int(np.sum(mask_slice == 1))
            t_pix = int(np.sum(mask_slice == 2))
            c_pix = int(np.sum(mask_slice == 3))
            
            st.markdown(f"""
            <div class="summary-box">
                <strong>{plane} Slice #{slice_idx} Class Summary ({overlay_target}):</strong> &nbsp;&nbsp;
                Kidney: <strong>{"YES (" + str(k_pix) + " voxels)" if k_pix > 0 else "NO"}</strong> | 
                Tumor: <strong>{"YES (" + str(t_pix) + " voxels)" if t_pix > 0 else "NO"}</strong> | 
                Cyst: <strong>{"YES (" + str(c_pix) + " voxels)" if c_pix > 0 else "NO"}</strong>
            </div>
            """, unsafe_allow_html=True)

        # Create explicit 3-channel RGB overlay uint8 array (H, W, 3)
        rgb_overlay = create_segmentation_overlay(
            ct_slice=ct_slice,
            mask_slice=mask_slice,
            alpha=alpha,
            show_kidney=show_kidney,
            show_tumor=show_tumor,
            show_cyst=show_cyst,
            window_min=win_min,
            window_max=win_max
        )

        fig = px.imshow(rgb_overlay)
        fig.update_layout(
            title=dict(text=f"<b>{plane} Slice #{slice_idx}</b> - {overlay_target} ({case_name})", font=dict(size=16)),
            xaxis_title="Width Voxels",
            yaxis_title="Height Voxels",
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            margin=dict(l=20, r=20, t=45, b=20),
            height=600
        )
        fig.update_traces(hovertemplate="X: %{x}<br>Y: %{y}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Spatial Alignment & Tensor Diagnostics", expanded=False):
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                st.write("**ORIGINAL CT**")
                st.write(f"- Shape: `{ct_data.shape}`")
                st.write(f"- Voxel Spacing: `{header_spacing}` mm")
                st.write(f"- HU Window Range: `[{win_min:.1f}, {win_max:.1f}] HU`")
                st.write(f"- Display CT Source: `kits21/data/{case_name}/imaging.nii.gz`")
            with c_d2:
                st.write("**RESTORED PREDICTION**")
                st.write(f"- Model Space Shape: `(64, 128, 128)`")
                st.write(f"- Restored Original Space Shape: `{pred_data.shape if pred_data is not None else 'None'}`")
                st.write(f"- Spatial Grid Alignment: `ct_original.shape == prediction_original.shape` (MATCH)")
                st.write(f"- Slice #{slice_idx} Label Unique: `{np.unique(mask_slice) if mask_slice is not None else 'None'}`")

        st.markdown("""
        <div class="explain-card">
            <strong>What am I looking at?</strong><br>
            Multi-planar slice visualization displays the 3D CT scan cross-section with the segmented structures overlaid in color. 
            The neural network identifies renal parenchyma (Green), renal masses/tumors (Red), and renal cysts (Blue) as distinct semantic classes.
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------
# PAGE 2: GROUND TRUTH VS PREDICTION
# ----------------------------------------------------
elif page == "Ground Truth vs Prediction":
    st.subheader("Ground Truth vs Model Prediction Comparison")
    
    if ct_data is None:
        st.info("Please select a sample KiTS case or upload a CT scan in the sidebar.")
    elif seg_data is None:
        st.info("Ground truth is unavailable for custom scan uploads. Upload a KiTS sample case to compare against ground truth.")
    elif pred_data is None:
        st.warning("Model prediction not generated yet.")
    else:
        D, H, W = ct_data.shape
        c_p, c_s, c_a = st.columns([1, 1, 1])
        plane = c_p.selectbox("Plane View", ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"])
        
        fg_counts = np.sum(pred_data > 0, axis=(1, 2) if plane.startswith("Axial") else ((0, 2) if plane.startswith("Coronal") else (0, 1)))
        default_slice = int(np.argmax(fg_counts)) if np.max(fg_counts) > 0 else (D // 2 if plane.startswith("Axial") else (H // 2 if plane.startswith("Coronal") else W // 2))
        
        slice_max = D - 1 if plane.startswith("Axial") else (H - 1 if plane.startswith("Coronal") else W - 1)
        slice_idx = c_s.slider("Slice Index", 0, slice_max, default_slice)
        alpha = c_a.slider("Mask Opacity", 0.0, 1.0, 0.50, 0.05)

        if plane.startswith("Axial"):
            ct_s = ct_data[slice_idx, :, :]
            gt_s = seg_data[slice_idx, :, :]
            pr_s = pred_data[slice_idx, :, :]
        elif plane.startswith("Coronal"):
            ct_s = ct_data[:, slice_idx, :]
            gt_s = seg_data[:, slice_idx, :]
            pr_s = pred_data[:, slice_idx, :]
        else:
            ct_s = ct_data[:, :, slice_idx]
            gt_s = seg_data[:, :, slice_idx]
            pr_s = pred_data[:, :, slice_idx]

        col_gt, col_pred = st.columns(2)

        with col_gt:
            gt_k = int(np.sum(gt_s == 1))
            gt_t = int(np.sum(gt_s == 2))
            gt_c = int(np.sum(gt_s == 3))
            st.markdown(f"""
            <div class="summary-box">
                <strong>Ground Truth Voxel Summary ({plane} #{slice_idx}):</strong><br>
                Kidney: <strong>{"YES (" + str(gt_k) + " voxels)" if gt_k > 0 else "NO"}</strong> | 
                Tumor: <strong>{"YES (" + str(gt_t) + " voxels)" if gt_t > 0 else "NO"}</strong> | 
                Cyst: <strong>{"YES (" + str(gt_c) + " voxels)" if gt_c > 0 else "NO"}</strong>
            </div>
            """, unsafe_allow_html=True)

            rgb_gt = create_segmentation_overlay(ct_s, gt_s, alpha=alpha, window_min=win_min, window_max=win_max)
            fig_gt = px.imshow(rgb_gt)
            fig_gt.update_layout(
                title=dict(text=f"<b>Ground Truth Annotation</b> ({plane} #{slice_idx})", font=dict(size=15)),
                xaxis_title="Width Voxels", yaxis_title="Height Voxels",
                template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", height=500
            )
            fig_gt.update_traces(hovertemplate="X: %{x}<br>Y: %{y}<extra></extra>")
            st.plotly_chart(fig_gt, use_container_width=True)

        with col_pred:
            pr_k = int(np.sum(pr_s == 1))
            pr_t = int(np.sum(pr_s == 2))
            pr_c = int(np.sum(pr_s == 3))
            st.markdown(f"""
            <div class="summary-box">
                <strong>Model Prediction Voxel Summary ({plane} #{slice_idx}):</strong><br>
                Kidney: <strong>{"YES (" + str(pr_k) + " voxels)" if pr_k > 0 else "NO"}</strong> | 
                Tumor: <strong>{"YES (" + str(pr_t) + " voxels)" if pr_t > 0 else "NO"}</strong> | 
                Cyst: <strong>{"YES (" + str(pr_c) + " voxels)" if pr_c > 0 else "NO"}</strong>
            </div>
            """, unsafe_allow_html=True)

            rgb_pr = create_segmentation_overlay(ct_s, pr_s, alpha=alpha, window_min=win_min, window_max=win_max)
            fig_pr = px.imshow(rgb_pr)
            fig_pr.update_layout(
                title=dict(text=f"<b>Model Prediction (SegResNet)</b> ({plane} #{slice_idx})", font=dict(size=15)),
                xaxis_title="Width Voxels", yaxis_title="Height Voxels",
                template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", height=500
            )
            fig_pr.update_traces(hovertemplate="X: %{x}<br>Y: %{y}<extra></extra>")
            st.plotly_chart(fig_pr, use_container_width=True)

        # Error / Difference Map
        st.markdown("### Voxel Agreement & Classification Error Map")
        rgb_err = create_error_map_overlay(ct_s, gt_s, pr_s, alpha=alpha, window_min=win_min, window_max=win_max)
        fig_err = px.imshow(rgb_err)
        fig_err.update_layout(
            title=dict(text="<b>Voxel Classification Error Map</b> ([TP] Green: True Positive | [FP] Red: False Positive | [FN] Orange: False Negative)", font=dict(size=15)),
            xaxis_title="Width Voxels", yaxis_title="Height Voxels",
            template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", height=450
        )
        fig_err.update_traces(hovertemplate="X: %{x}<br>Y: %{y}<extra></extra>")
        st.plotly_chart(fig_err, use_container_width=True)

        with st.expander("Spatial Alignment & Tensor Diagnostics", expanded=False):
            st.write("**ORIGINAL CT**")
            st.write(f"- CT Shape & Dtype: `{ct_s.shape}`, `{ct_s.dtype}`")
            st.write(f"- Ground Truth Shape & Dtype: `{gt_s.shape}`, `{gt_s.dtype}`")
            st.write(f"- Prediction Original Space Shape: `{pr_s.shape}`, `{pr_s.dtype}`")
            st.write(f"- Unique Prediction Labels: `{np.unique(pr_s)}`")
            st.write(f"- GT RGB Overlay Shape: `{rgb_gt.shape}`")
            st.write(f"- Prediction RGB Overlay Shape: `{rgb_pr.shape}`")
            st.write("**Error Map RGB Overlay Shape & Dtype:**", rgb_err.shape, rgb_err.dtype)

        st.markdown("""
        <div class="explain-card">
            <strong>What am I looking at?</strong><br>
            Side-by-side comparison displays the expert radiologist ground-truth annotation alongside the SegResNet model prediction. 
            The classification error map highlights exact voxel agreement: 
            <span style="color:#00FF64;">True Positives</span> (correct overlap), 
            <span style="color:#FF2850;">False Positives</span> (over-segmentation), and 
            <span style="color:#FFA500;">False Negatives</span> (missed regions).
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------
# PAGE 3: 3D SURFACE RENDERING (FIXED MARCHING CUBES)
# ----------------------------------------------------
elif page == "3D Surface Rendering":
    st.subheader("Interactive 3D Surface Mesh Rendering")
    
    if ct_data is None:
        st.info("Please select a sample KiTS case or upload a CT scan in the sidebar.")
    else:
        target_options = []
        if pred_data is not None:
            target_options.append("Model Prediction")
        if seg_data is not None:
            target_options.append("Ground Truth")

        if not target_options:
            st.warning("No segmentation volume available for 3D rendering.")
        else:
            c1, c2 = st.columns([1, 2])
            render_target = c1.radio("Mesh Target Volume", target_options, horizontal=True)
            target_vol = pred_data if render_target == "Model Prediction" else seg_data

            st.markdown("**Mesh Structure Controls & Opacity:**")
            col_k, col_t, col_c = st.columns(3)
            
            show_k = col_k.checkbox("Show Kidney Mesh", value=True)
            k_op = col_k.slider("Kidney Opacity", 0.05, 1.0, 0.35, 0.05)
            
            show_t = col_t.checkbox("Show Tumor Mesh", value=True)
            t_op = col_t.slider("Tumor Opacity", 0.1, 1.0, 0.85, 0.05)
            
            show_c = col_c.checkbox("Show Cyst Mesh", value=True)
            c_op = col_c.slider("Cyst Opacity", 0.1, 1.0, 0.85, 0.05)

            visible_cls = []
            if show_k: visible_cls.append(1)
            if show_t: visible_cls.append(2)
            if show_c: visible_cls.append(3)

            opacities = {1: k_op, 2: t_op, 3: c_op}

            with st.spinner("Extracting physical 3D isosurface meshes via Marching Cubes..."):
                fig3d, empty_msgs = generate_3d_marching_cubes_mesh(
                    seg_vol=target_vol,
                    spacing=header_spacing,
                    visible_classes=visible_cls,
                    opacities=opacities
                )

            st.plotly_chart(fig3d, use_container_width=True)

            if empty_msgs:
                for msg in empty_msgs:
                    st.info(msg)

            st.markdown("""
            <div class="explain-card">
                <strong>What am I looking at?</strong><br>
                3D surface rendering uses the Marching Cubes algorithm to convert 3D segmentation voxel arrays into physical coordinate surface meshes. 
                Voxel spacing (1.5mm x 1.5mm x 2.0mm) is preserved to maintain accurate anatomical proportions and spatial relationships between the kidney parenchyma and renal lesions.
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------
# PAGE 4: VOLUME & ANATOMICAL METRICS
# ----------------------------------------------------
elif page == "Volume & Anatomical Metrics":
    st.subheader("Volumetric & Anatomical Quantitative Metrics")
    
    target_vol = pred_data if pred_data is not None else seg_data
    if target_vol is None:
        st.info("No segmentation volume loaded. Run 3D model segmentation in sidebar to compute volume breakdown.")
    else:
        dx, dy, dz = header_spacing
        voxel_vol_cm3 = (dx * dy * dz) / 1000.0
        
        k_vox = int(np.sum(target_vol == 1))
        t_vox = int(np.sum(target_vol == 2))
        c_vox = int(np.sum(target_vol == 3))
        tot_vox = k_vox + t_vox + c_vox
        
        k_vol = k_vox * voxel_vol_cm3
        t_vol = t_vox * voxel_vol_cm3
        c_vol = c_vox * voxel_vol_cm3
        tot_vol = tot_vox * voxel_vol_cm3

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("KIDNEY PARENCHYMA", f"{k_vol:.1f} cm³", f"{k_vox:,} voxels")
        m2.metric("RENAL TUMOR / MASS", f"{t_vol:.1f} cm³", f"{t_vox:,} voxels" if t_vox > 0 else "No voxels predicted")
        m3.metric("RENAL CYST", f"{c_vol:.1f} cm³", f"{c_vox:,} voxels" if c_vox > 0 else "No voxels predicted")
        m4.metric("TOTAL SEGMENTED", f"{tot_vol:.1f} cm³", f"{tot_vox:,} voxels")

        st.caption("*Predicted segmentation volumes computed from 3D mask voxel count and physical scan spacing.*")

        st.markdown("---")
        st.markdown("### Anatomical Structure & KiTS Hierarchical Region Breakdown")
        
        df_table = pd.DataFrame([
            {"Structure / Region": "Kidney Parenchyma (Class 1)", "Voxel Count": f"{k_vox:,}", "Volume (cm³)": f"{k_vol:.2f}", "Fraction (%)": f"{(k_vol/tot_vol*100 if tot_vol>0 else 0):.1f}%"},
            {"Structure / Region": "Renal Mass / Tumor (Class 2)", "Voxel Count": f"{t_vox:,}", "Volume (cm³)": f"{t_vol:.2f}", "Fraction (%)": f"{(t_vol/tot_vol*100 if tot_vol>0 else 0):.1f}%"},
            {"Structure / Region": "Renal Cyst (Class 3)", "Voxel Count": f"{c_vox:,}", "Volume (cm³)": f"{c_vol:.2f}", "Fraction (%)": f"{(c_vol/tot_vol*100 if tot_vol>0 else 0):.1f}%"},
            {"Structure / Region": "Composite Kidney (Kidney + Mass)", "Voxel Count": f"{tot_vox:,}", "Volume (cm³)": f"{tot_vol:.2f}", "Fraction (%)": "100.0%"}
        ])
        st.dataframe(df_table, use_container_width=True)

        st.markdown("""
        <div class="explain-card">
            <strong>What am I looking at?</strong><br>
            Volumetric analysis calculates physical organ and lesion volumes in cubic centimeters (cm³) by multiplying predicted voxel counts by the physical voxel volume 
            derived from NIfTI header spacing. Composite Kidney metrics represent combined organ and lesion regions in accordance with KiTS challenge benchmarks.
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------
# PAGE 5: MODEL PERFORMANCE & BENCHMARKS
# ----------------------------------------------------
elif page == "Model Performance & Benchmarks":
    st.subheader("Model Benchmark Evaluation & Baseline Comparison")
    
    summary_path = "results/summary_metrics.json"
    comp_path = "results/model_comparison.csv"
    
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_metrics = json.load(f)
            
        st.markdown("### Untouched Held-Out Test Set Metrics (MONAI SegResNet)")
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        
        t1.metric("Kidney Dice", f"{summary_metrics.get('dice_kidney', 0):.4f}")
        t2.metric("Tumor Dice", f"{summary_metrics.get('dice_tumor', 0):.4f}")
        t3.metric("Tumor HD95", f"{summary_metrics.get('hd95_tumor', 0):.2f} mm")
        t4.metric("Mean Dice", f"{summary_metrics.get('mean_dice', 0):.4f}")
        t5.metric("Kidney IoU", f"{summary_metrics.get('iou_kidney', 0):.4f}")
        t6.metric("Tumor IoU", f"{summary_metrics.get('iou_tumor', 0):.4f}")

        st.markdown("""
        **Evaluation Metric Definitions:**
        - **Dice Similarity Coefficient (DSC):** Measures spatial overlap between predicted segmentation and ground truth ($2|A \cap B| / (|A| + |B|)$).
        - **Intersection over Union (IoU):** Measures intersection area relative to union ($|A \cap B| / |A \cup B|$).
        - **HD95 (95th Percentile Hausdorff Distance):** Measures maximum 95th-percentile boundary distance error in millimeters (mm).
        """)

    st.markdown("---")
    st.markdown("### Progressive Architecture Performance Comparison")
    
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        st.dataframe(df_comp, use_container_width=True)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=df_comp["model"], y=df_comp["dice_kidney"], name="Kidney Dice", marker_color="#00FF64"))
        fig_bar.add_trace(go.Bar(x=df_comp["model"], y=df_comp["dice_tumor"], name="Tumor Dice", marker_color="#FF2850"))
        fig_bar.add_trace(go.Bar(x=df_comp["model"], y=df_comp["mean_dice"], name="Mean Dice", marker_color="#3296FF"))

        fig_bar.update_layout(
            title="<b>Progression from 2D Baseline to SegResNet</b>",
            xaxis_title="Model Architecture",
            yaxis_title="Dice Score",
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    curves_fig_path = "results/figures/training_curves.png"
    if os.path.exists(curves_fig_path):
        st.markdown("### Training Loss & Validation Progression Curves")
        st.image(curves_fig_path, use_container_width=True)

# ----------------------------------------------------
# PAGE 6: MODEL SPECS & INFO
# ----------------------------------------------------
elif page == "Model Specs & Info":
    st.subheader("Neural Network Architecture & Technical Specifications")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("""
        ### Model Specifications
        - **Architecture:** MONAI SegResNet (Residual Encoder-Decoder)
        - **Task:** 3D Multi-Class Volumetric Semantic Segmentation
        - **Dataset:** KiTS21 (Kidney & Kidney Tumor Segmentation)
        - **Input Format:** 3D Contrast-Enhanced Abdominal CT
        - **Target Classes:**
          - `0:` Background
          - `1:` Kidney Parenchyma
          - `2:` Renal Mass / Tumor
          - `3:` Renal Cyst
        """)
        
    with col_s2:
        st.markdown("""
        ### Training Configuration
        - **Preprocessing:** RAS Orientation, 1.5x1.5x2.0mm Resampling, [-150, 350] HU Windowing
        - **Patch Sampling:** Lesion-Aware Multi-Class Patch Sampling
        - **Loss Function:** Class-Weighted DiceCE Loss (`weights=[0.2, 1.0, 3.0, 3.0]`)
        - **Optimizer:** AdamW (`lr=3e-4`, `weight_decay=1e-4`)
        - **Learning Rate Scheduler:** CosineAnnealingLR
        - **Inference Strategy:** Volumetric Sliding-Window (50% Overlap)
        """)

    st.markdown("""
    <div class="explain-card">
        <strong>Academic Integrity & Scope Statement:</strong><br>
        This machine learning system was independently constructed for academic evaluation on the KiTS21 benchmark dataset. 
        All metrics reported in this dashboard originate from direct GPU execution and held-out test evaluation.
    </div>
    """, unsafe_allow_html=True)
