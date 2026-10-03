import os
import json
import glob
import pandas as pd
import numpy as np
import nibabel as nib
from typing import Dict, List, Any
from src.utils.logger import get_logger
from src.data.label_mapping import verify_and_map_labels

logger = get_logger("DatasetAudit")

def audit_case(case_dir: str, preferred_seg_name: str = "aggregated_MAJ_seg.nii.gz") -> Dict[str, Any]:
    """
    Audits a single KiTS21 case directory.
    """
    case_id = os.path.basename(case_dir.rstrip("/\\"))
    imaging_path = os.path.join(case_dir, "imaging.nii.gz")
    
    # Try preferred segmentation file, fallback to aggregated_OR_seg or segmentations folder if needed
    seg_path = os.path.join(case_dir, preferred_seg_name)
    if not os.path.exists(seg_path):
        fallback_seg = os.path.join(case_dir, "aggregated_OR_seg.nii.gz")
        if os.path.exists(fallback_seg):
            seg_path = fallback_seg
        else:
            seg_path = None

    record = {
        "case_id": case_id,
        "imaging_exists": os.path.exists(imaging_path),
        "seg_exists": seg_path is not None and os.path.exists(seg_path),
        "seg_filename": os.path.basename(seg_path) if seg_path else None,
        "corrupt": False,
        "error_message": "",
        "shape": None,
        "spacing": None,
        "dtype": None,
        "min_hu": None,
        "max_hu": None,
        "mean_hu": None,
        "std_hu": None,
        "label_values": [],
        "kidney_voxels": 0,
        "tumor_voxels": 0,
        "cyst_voxels": 0,
        "kidney_present": False,
        "tumor_present": False,
        "cyst_present": False
    }

    if not record["imaging_exists"]:
        record["corrupt"] = True
        record["error_message"] = "Imaging NIfTI file missing."
        return record

    try:
        # Inspect CT imaging file
        img_nifti = nib.load(imaging_path)
        img_data = img_nifti.get_fdata(caching='unchanged')
        header = img_nifti.header
        
        record["shape"] = list(img_data.shape)
        record["spacing"] = [float(x) for x in header.get_zooms()[:3]]
        record["dtype"] = str(img_data.dtype)
        record["min_hu"] = float(np.min(img_data))
        record["max_hu"] = float(np.max(img_data))
        record["mean_hu"] = float(np.mean(img_data))
        record["std_hu"] = float(np.std(img_data))

        # Inspect segmentation file if present
        if record["seg_exists"]:
            seg_nifti = nib.load(seg_path)
            seg_data = seg_nifti.get_fdata(caching='unchanged')
            mapped_seg, label_stats = verify_and_map_labels(seg_data)
            
            record["label_values"] = [int(x) for x in np.unique(seg_data)]
            record["kidney_voxels"] = int(np.sum(mapped_seg == 1))
            record["tumor_voxels"] = int(np.sum(mapped_seg == 2))
            record["cyst_voxels"] = int(np.sum(mapped_seg == 3))
            
            record["kidney_present"] = record["kidney_voxels"] > 0
            record["tumor_present"] = record["tumor_voxels"] > 0
            record["cyst_present"] = record["cyst_voxels"] > 0
        else:
            record["error_message"] = "Segmentation NIfTI file missing."

    except Exception as e:
        record["corrupt"] = True
        record["error_message"] = str(e)
        logger.error(f"Error auditing case {case_id}: {e}")

    return record

def run_dataset_audit(data_dir: str, output_dir: str = "results") -> Dict[str, Any]:
    """
    Audits all available KiTS21 cases in data_dir and generates summary CSV & JSON artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    case_dirs = sorted(glob.glob(os.path.join(data_dir, "case_*")))
    logger.info(f"Found {len(case_dirs)} cases in {data_dir}")

    audit_records = []
    for cdir in case_dirs:
        record = audit_case(cdir)
        audit_records.append(record)

    df = pd.DataFrame(audit_records)
    
    # Save CSV summary
    csv_path = os.path.join(output_dir, "dataset_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved dataset summary to {csv_path}")

    # Generate summary metrics
    valid_df = df[~df["corrupt"]]
    summary_dict = {
        "total_cases_found": len(case_dirs),
        "valid_cases": len(valid_df),
        "corrupt_or_missing_cases": len(df) - len(valid_df),
        "cases_with_kidney": int(valid_df["kidney_present"].sum()) if len(valid_df) > 0 else 0,
        "cases_with_tumor": int(valid_df["tumor_present"].sum()) if len(valid_df) > 0 else 0,
        "cases_with_cyst": int(valid_df["cyst_present"].sum()) if len(valid_df) > 0 else 0,
        "mean_shape": valid_df["shape"].apply(lambda x: str(x)).mode().iloc[0] if len(valid_df) > 0 else None,
        "audit_records": audit_records
    }

    # Save JSON summary
    json_path = os.path.join(output_dir, "dataset_audit.json")
    with open(json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Saved dataset audit JSON to {json_path}")

    return summary_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="kits21/data")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    
    run_dataset_audit(args.data_dir, args.output_dir)
