import os
import sys
import numpy as np
import nibabel as nib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger

logger = get_logger("SampleDataGen")

def create_sample_case(case_dir: str, shape=(64, 128, 128)):
    os.makedirs(case_dir, exist_ok=True)
    img_path = os.path.join(case_dir, "imaging.nii.gz")
    seg_path = os.path.join(case_dir, "aggregated_MAJ_seg.nii.gz")

    if os.path.exists(img_path) and os.path.exists(seg_path):
        return

    D, H, W = shape
    # CT intensity in HU: background ~ -1000, soft tissue ~ 40, kidney ~ 120, contrast tumor ~ 180
    ct_data = np.random.normal(-100, 50, (D, H, W)).astype(np.float32)
    seg_data = np.zeros((D, H, W), dtype=np.uint8)

    # Left Kidney
    ct_data[15:45, 30:60, 30:60] += 150
    seg_data[15:45, 30:60, 30:60] = 1

    # Tumor inside Left Kidney
    ct_data[22:38, 40:55, 40:55] += 60
    seg_data[22:38, 40:55, 40:55] = 2

    # Right Kidney
    ct_data[15:45, 30:60, 68:98] += 150
    seg_data[15:45, 30:60, 68:98] = 1

    # Cyst inside Right Kidney
    ct_data[25:35, 42:52, 75:85] -= 100
    seg_data[25:35, 42:52, 75:85] = 3

    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    img_nifti = nib.Nifti1Image(ct_data, affine)
    seg_nifti = nib.Nifti1Image(seg_data, affine)

    nib.save(img_nifti, img_path)
    nib.save(seg_nifti, seg_path)

def generate_all_samples(output_dir="kits21/data", num_cases=10):
    logger.info(f"Generating {num_cases} sample KiTS21 CT volumes in {output_dir}...")
    for i in range(num_cases):
        case_id = f"case_{i:05d}"
        cdir = os.path.join(output_dir, case_id)
        create_sample_case(cdir)
    logger.info("Sample CT volumes generated successfully!")

if __name__ == "__main__":
    generate_all_samples()
