import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any
from monai.data import Dataset as MONAIDataset, DataLoader as MONAIDataLoader, CacheDataset
from src.utils.logger import get_logger
from src.data.preprocessing import get_train_transforms, get_val_transforms

logger = get_logger("Dataset")

def get_case_paths(data_dir: str, case_ids: List[str], preferred_seg_name: str = "aggregated_MAJ_seg.nii.gz") -> List[Dict[str, str]]:
    """
    Constructs image and label file path dictionaries for MONAI datasets.
    """
    data_dicts = []
    missing_count = 0
    
    for case_id in case_ids:
        case_dir = os.path.join(data_dir, case_id)
        img_path = os.path.join(case_dir, "imaging.nii.gz")
        
        # Check segmentation file
        seg_path = os.path.join(case_dir, preferred_seg_name)
        if not os.path.exists(seg_path):
            fallback = os.path.join(case_dir, "aggregated_OR_seg.nii.gz")
            if os.path.exists(fallback):
                seg_path = fallback
            else:
                seg_path = None
                
        if os.path.exists(img_path) and seg_path and os.path.exists(seg_path):
            data_dicts.append({"image": img_path, "label": seg_path, "case_id": case_id})
        else:
            missing_count += 1
            logger.warning(f"Case {case_id} missing imaging or seg file! Skipping in loader.")
            
    logger.info(f"Loaded {len(data_dicts)} case paths (skipped {missing_count} missing/invalid).")
    return data_dicts

def create_dataloaders(
    data_dir: str,
    split_dir: str = "data/splits",
    batch_size: int = 2,
    num_workers: int = 0,
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    cache_rate: float = 0.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates Train, Validation, and Test DataLoaders.
    """
    train_csv = os.path.join(split_dir, "train.csv")
    val_csv = os.path.join(split_dir, "val.csv")
    test_csv = os.path.join(split_dir, "test.csv")
    
    train_ids = pd.read_csv(train_csv)["case_id"].tolist()
    val_ids = pd.read_csv(val_csv)["case_id"].tolist()
    test_ids = pd.read_csv(test_csv)["case_id"].tolist()
    
    train_files = get_case_paths(data_dir, train_ids)
    val_files = get_case_paths(data_dir, val_ids)
    test_files = get_case_paths(data_dir, test_ids)
    
    train_transforms = get_train_transforms(patch_size=patch_size)
    val_transforms = get_val_transforms()
    
    if cache_rate > 0.0:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    else:
        train_ds = MONAIDataset(data=train_files, transform=train_transforms)
        val_ds = MONAIDataset(data=val_files, transform=val_transforms)
        
    test_ds = MONAIDataset(data=test_files, transform=val_transforms)
    
    train_loader = MONAIDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = MONAIDataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = MONAIDataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    return train_loader, val_loader, test_loader
