import torch
from typing import Dict, Tuple, List, Optional
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    CastToTyped,
    EnsureTyped
)

# KiTS Abdominal CT Windowing Parameters (-150 to 350 HU)
DEFAULT_HU_MIN = -150.0
DEFAULT_HU_MAX = 350.0
DEFAULT_TARGET_SPACING = (1.5, 1.5, 2.0)

def get_train_transforms(
    patch_size: Tuple[int, int, int] = (48, 64, 64),
    target_spacing: Tuple[float, float, float] = DEFAULT_TARGET_SPACING,
    hu_min: float = DEFAULT_HU_MIN,
    hu_max: float = DEFAULT_HU_MAX,
    num_samples: int = 4,
    class_ratios: Tuple[float, float, float, float] = (0.1, 0.4, 0.3, 0.2)
) -> Compose:
    """
    Constructs the enhanced training data transformation pipeline featuring
    Lesion-Aware Multi-Class Patch Sampling and Realistic 3D Medical Augmentations.
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=target_spacing,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10, allow_smaller=True),
        # Lesion-Aware Multi-Class Patch Sampling
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            num_classes=4,
            ratios=list(class_ratios),
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
            allow_smaller=True
        ),
        # 3D Spatial & Intensity Augmentations
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotated(
            keys=["image", "label"],
            range_x=0.3,
            range_y=0.3,
            range_z=0.3,
            mode=("bilinear", "nearest"),
            prob=0.4
        ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.85,
            max_zoom=1.15,
            mode=("bilinear", "nearest"),
            prob=0.3
        ),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=["image"], std=0.05, prob=0.2),
        RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.5), prob=0.2),
        CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
        EnsureTyped(keys=["image", "label"])
    ])

def get_val_transforms(
    target_spacing: Tuple[float, float, float] = DEFAULT_TARGET_SPACING,
    hu_min: float = DEFAULT_HU_MIN,
    hu_max: float = DEFAULT_HU_MAX
) -> Compose:
    """
    Constructs deterministic validation/test preprocessing pipeline.
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=target_spacing,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10, allow_smaller=True),
        CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
        EnsureTyped(keys=["image", "label"])
    ])

def get_inference_transforms(
    target_spacing: Tuple[float, float, float] = DEFAULT_TARGET_SPACING,
    hu_min: float = DEFAULT_HU_MIN,
    hu_max: float = DEFAULT_HU_MAX
) -> Compose:
    """
    Constructs inference preprocessing pipeline for single CT imaging file.
    """
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=target_spacing,
            mode="bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CastToTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["image"])
    ])
