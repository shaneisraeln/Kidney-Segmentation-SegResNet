import os
import torch
import numpy as np
import nibabel as nib
from typing import Dict, Any, Optional, Tuple
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CastToTyped, EnsureTyped
from monai.inferers import sliding_window_inference
from src.utils.logger import get_logger
from src.models import get_model
from src.data.preprocessing import DEFAULT_HU_MIN, DEFAULT_HU_MAX, DEFAULT_TARGET_SPACING

logger = get_logger("Inference")

def predict_ct_volume(
    model: torch.nn.Module,
    ct_path: str,
    device: torch.device,
    patch_size: tuple = (96, 96, 96),
    overlap: float = 0.5,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Runs sliding-window inference on a single 3D CT scan, preserving original affine metadata.
    
    Returns:
        pred_mask_np: 3D segmentation mask (D, H, W) numpy array
        nifti_obj: NiBabel NIfTI object containing affine metadata
    """
    model.eval()
    model.to(device)

    # Load original NIfTI header for output metadata preservation
    orig_nifti = nib.load(ct_path)
    orig_affine = orig_nifti.affine
    orig_header = orig_nifti.header
    orig_shape = orig_nifti.shape

    # Determine orientation transform from original NIfTI orientation to RAS space and back
    orig_axcodes = nib.orientations.aff2axcodes(orig_affine)
    orig_ornt = nib.orientations.axcodes2ornt(orig_axcodes)
    ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    
    to_ras = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    to_orig = nib.orientations.ornt_transform(ras_ornt, orig_ornt)

    # Determine target shape in RAS space for 3D resampling
    dummy_ones = np.ones(orig_shape, dtype=np.uint8)
    ras_target_shape = nib.orientations.apply_orientation(dummy_ones, to_ras).shape

    val_transforms = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=DEFAULT_TARGET_SPACING, mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=DEFAULT_HU_MIN, a_max=DEFAULT_HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CastToTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["image"])
    ])

    sample = val_transforms({"image": ct_path})
    image_tensor = sample["image"].unsqueeze(0).to(device) # (1, C, D, H, W)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = sliding_window_inference(
                inputs=image_tensor,
                roi_size=patch_size,
                sw_batch_size=2,
                predictor=model,
                overlap=overlap
            )
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).squeeze(0) # (D, H, W) in RAS space

    preds_4d = preds.cpu().float().unsqueeze(0).unsqueeze(0)

    # 1. 3D Nearest-Neighbor Spatial Resampling in RAS Space
    if preds_4d.shape[2:] != ras_target_shape:
        preds_ras = torch.nn.functional.interpolate(
            preds_4d,
            size=ras_target_shape,
            mode="nearest"
        ).squeeze(0).squeeze(0).byte().numpy()
    else:
        preds_ras = preds.cpu().numpy().astype(np.uint8)

    # 2. Invert Orientation from RAS back to Original NIfTI Orientation
    pred_mask_np = nib.orientations.apply_orientation(preds_ras, to_orig).astype(np.uint8)

    # Strict spatial shape alignment assertion
    assert pred_mask_np.shape == orig_shape, f"Restored prediction shape {pred_mask_np.shape} != original CT shape {orig_shape}"

    # Re-construct output NIfTI with original shape, affine, and header
    out_nifti = nib.Nifti1Image(pred_mask_np, affine=orig_affine, header=orig_header)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(out_nifti, output_path)
        logger.info(f"Saved prediction NIfTI (Original Space {orig_shape}) to: {output_path}")

    return pred_mask_np, out_nifti

def load_trained_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads model architecture and trained weights from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    
    model_name = model_cfg.get("name", "unet3d")
    in_channels = model_cfg.get("in_channels", 1)
    out_channels = model_cfg.get("out_channels", 4)
    
    model = get_model(model_name, in_channels=in_channels, out_channels=out_channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Successfully loaded {model_name} from {checkpoint_path}")
    return model
