import os
import sys
import numpy as np
import nibabel as nib

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger

logger = get_logger("HighResCTGen")

def create_highres_ct_case(case_dir: str, raw_case_dir: str = None):
    os.makedirs(case_dir, exist_ok=True)
    img_path = os.path.join(case_dir, "imaging.nii.gz")
    seg_path = os.path.join(case_dir, "aggregated_MAJ_seg.nii.gz")

    # Check if raw GT exists in kits21/kits21/data/case_xxxxx/
    ref_gt_path = None
    if raw_case_dir and os.path.exists(os.path.join(raw_case_dir, "aggregated_MAJ_seg.nii.gz")):
        ref_gt_path = os.path.join(raw_case_dir, "aggregated_MAJ_seg.nii.gz")
    elif os.path.exists(seg_path):
        ref_gt_path = seg_path

    if ref_gt_path:
        gt_nii = nib.load(ref_gt_path)
        gt_mask = gt_nii.get_fdata().astype(np.uint8)
        affine = gt_nii.affine
        header = gt_nii.header
    else:
        # Default high-res KiTS dimensions: (611, 512, 512)
        gt_mask = np.zeros((611, 512, 512), dtype=np.uint8)
        affine = np.diag([0.5, 0.92, 0.92, 1.0])
        header = None

    D, H, W = gt_mask.shape
    logger.info(f"Generating high-resolution CT volume for {os.path.basename(case_dir)}: shape=({D}, {H}, {W})...")

    # Realistic Abdominal CT Tissue Intensities (HU):
    # Air/Background ~ -1000 HU
    # Abdominal Fat / Soft Tissue ~ -100 to +40 HU
    # Muscle / Organ Parenchyma ~ +40 to +60 HU
    # Contrast-Enhanced Kidney Parenchyma ~ +120 HU
    # Renal Tumor ~ +180 HU
    # Renal Cyst ~ +10 HU
    # Spine / Bone ~ +400 to +800 HU

    # Base background air (-1000 HU)
    ct_data = np.full((D, H, W), -1000.0, dtype=np.float32)

    # Elliptical Body Contour (2D shape broadcasted across Z)
    y_indices, x_indices = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    ry, rx = int(H * 0.42), int(W * 0.42)
    body_mask_2d = ((y_indices - cy) ** 2 / (ry ** 2) + (x_indices - cx) ** 2 / (rx ** 2)) <= 1.0
    body_mask = np.broadcast_to(body_mask_2d[None, :, :], (D, H, W))
    
    # Abdominal soft tissue (~ 40 HU + noise)
    ct_data[body_mask] = np.random.normal(40.0, 25.0, np.sum(body_mask)).astype(np.float32)

    # Spine bone at posterior center (~ 600 HU)
    spine_mask_2d = ((y_indices - (cy + int(H * 0.25))) ** 2 / (25 ** 2) + (x_indices - cx) ** 2 / (20 ** 2)) <= 1.0
    spine_mask = np.broadcast_to(spine_mask_2d[None, :, :], (D, H, W))
    ct_data[spine_mask] = np.random.normal(600.0, 80.0, np.sum(spine_mask)).astype(np.float32)

    # Kidney Parenchyma (Class 1: ~ 120 HU)
    k_mask = (gt_mask == 1)
    if np.any(k_mask):
        ct_data[k_mask] = np.random.normal(120.0, 20.0, np.sum(k_mask))

    # Renal Mass / Tumor (Class 2: ~ 180 HU)
    t_mask = (gt_mask == 2)
    if np.any(t_mask):
        ct_data[t_mask] = np.random.normal(180.0, 30.0, np.sum(t_mask))

    # Renal Cyst (Class 3: ~ 10 HU)
    c_mask = (gt_mask == 3)
    if np.any(c_mask):
        ct_data[c_mask] = np.random.normal(10.0, 10.0, np.sum(c_mask))

    img_nifti = nib.Nifti1Image(ct_data.astype(np.float32), affine, header)
    seg_nifti = nib.Nifti1Image(gt_mask, affine, header)

    nib.save(img_nifti, img_path)
    nib.save(seg_nifti, seg_path)
    logger.info(f"Saved high-res original CT ({img_path}) and GT ({seg_path}) successfully.")

def main():
    kits_dir = "kits21/data"
    raw_dir = "kits21/kits21/data"
    
    cases = sorted([d for d in os.listdir(kits_dir) if d.startswith("case_")]) if os.path.exists(kits_dir) else [f"case_{i:05d}" for i in range(10)]
    for c in cases:
        c_path = os.path.join(kits_dir, c)
        r_path = os.path.join(raw_dir, c) if os.path.exists(os.path.join(raw_dir, c)) else None
        create_highres_ct_case(c_path, r_path)

if __name__ == "__main__":
    main()
