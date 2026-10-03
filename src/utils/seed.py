import random
import os
import numpy as np
import torch
import monai.utils.misc
import monai.transforms.compose
import monai.transforms.transform
import monai.data.utils
from monai.utils import set_determinism, convert_data_type
from monai.data.utils import to_affine_nd, AFFINE_TOL

# Monkeypatch NumPy 2.0 ptp incompatibility inside MONAI compute_shape_offset
def patched_compute_shape_offset(
    spatial_shape,
    in_affine,
    out_affine,
    scale_extent: bool = False,
):
    shape = np.array(spatial_shape, copy=True, dtype=float)
    sr = len(shape)
    in_affine_ = convert_data_type(to_affine_nd(sr, in_affine), np.ndarray)[0]
    out_affine_ = convert_data_type(to_affine_nd(sr, out_affine), np.ndarray)[0]
    in_coords = [(-0.5, dim - 0.5) if scale_extent else (0.0, dim - 1.0) for dim in shape]
    corners: np.ndarray = np.asarray(np.meshgrid(*in_coords, indexing="ij")).reshape((len(shape), -1))
    corners = np.concatenate((corners, np.ones_like(corners[:1])))
    try:
        corners_out = np.linalg.solve(out_affine_, in_affine_) @ corners
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Affine {out_affine_} is not invertible") from e
    corners = in_affine_ @ corners
    all_dist = corners_out[:-1].copy()
    corners_out = corners_out[:-1] / corners_out[-1]
    
    # NumPy 2.0 fix: np.ptp(arr, axis=1) instead of corners_out.ptp(axis=1)
    ptp_vals = np.ptp(corners_out, axis=1)
    out_shape = np.round(ptp_vals) if scale_extent else np.round(ptp_vals + 1.0)
    
    offset = None
    for i in range(corners.shape[1]):
        min_corner = np.min(all_dist - all_dist[:, i : i + 1], 1)
        if np.allclose(min_corner, 0.0, rtol=AFFINE_TOL):
            offset = corners[:-1, i]
            break
    if offset is None:
        offset = in_affine_[:-1, :-1] @ (shape / 2.0) + in_affine_[:-1, -1] - out_affine_[:-1, :-1] @ (out_shape / 2.0)
    if scale_extent:
        in_offset = np.append(0.5 * (shape / out_shape - 1.0), 1.0)
        offset = np.abs((in_affine_ @ in_offset / in_offset[-1])[:-1]) * np.sign(offset)
    return out_shape.astype(int, copy=False), offset

# Apply monkeypatch across MONAI modules
monai.data.utils.compute_shape_offset = patched_compute_shape_offset
import monai.transforms.spatial.array
monai.transforms.spatial.array.compute_shape_offset = patched_compute_shape_offset

def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, PyTorch, and MONAI.
    """
    safe_seed = int(seed) % 2147483647
    monai.utils.misc.MAX_SEED = 2147483647
    monai.transforms.compose.MAX_SEED = 2147483647
    monai.transforms.transform.MAX_SEED = 2147483647
    monai.data.utils.MAX_SEED = 2147483647
    random.seed(safe_seed)
    os.environ['PYTHONHASHSEED'] = str(safe_seed)
    np.random.seed(safe_seed)
    torch.manual_seed(safe_seed)
    torch.cuda.manual_seed(safe_seed)
    torch.cuda.manual_seed_all(safe_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=safe_seed)
    print(f"[SEED] Set all random seeds to: {safe_seed}")
