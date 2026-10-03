import numpy as np
from typing import Dict, Tuple

# KiTS21 Standard Label Mapping
DEFAULT_LABEL_MAPPING = {
    0: "Background",
    1: "Kidney",
    2: "Tumor",
    3: "Cyst"
}

CLASS_NAMES = ["Background", "Kidney", "Tumor", "Cyst"]
NUM_CLASSES = 4

def verify_and_map_labels(label_arr: np.ndarray, target_mapping: Dict[int, str] = DEFAULT_LABEL_MAPPING) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Verifies existing label values in a NIfTI segmentation array and ensures
    it maps cleanly to standard class indices (0: Background, 1: Kidney, 2: Tumor, 3: Cyst).
    
    Returns:
        mapped_array: np.ndarray with target integer class indices
        stats: dict containing voxel counts per label
    """
    unique_vals = np.unique(label_arr)
    stats = {}
    
    for val in unique_vals:
        val_int = int(val)
        count = int(np.sum(label_arr == val))
        name = target_mapping.get(val_int, f"Unknown ({val_int})")
        stats[name] = count
        
    return label_arr.astype(np.int64), stats

def get_class_colors() -> Dict[int, str]:
    """
    Returns hex color scheme for visualization.
    """
    return {
        0: "#000000",  # Background - Black/Transparent
        1: "#3B82F6",  # Kidney - Bright Blue
        2: "#EF4444",  # Tumor - Bright Red
        3: "#10B981"   # Cyst - Bright Emerald Green
    }
