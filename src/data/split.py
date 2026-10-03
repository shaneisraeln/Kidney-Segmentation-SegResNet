import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from src.utils.logger import get_logger

logger = get_logger("DatasetSplit")

def create_patient_splits(
    case_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_dir: str = "data/splits"
) -> Tuple[List[str], List[str], List[str]]:
    """
    Creates reproducible, leakage-free patient-level splits for KiTS21 cases.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort for deterministic shuffling
    sorted_cases = sorted(list(set(case_ids)))
    
    rng = np.random.RandomState(seed)
    shuffled_cases = sorted_cases.copy()
    rng.shuffle(shuffled_cases)
    
    n_total = len(shuffled_cases)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_cases = shuffled_cases[:n_train]
    val_cases = shuffled_cases[n_train:n_train + n_val]
    test_cases = shuffled_cases[n_train + n_val:]
    
    # Run Leakage Check
    train_set = set(train_cases)
    val_set = set(val_cases)
    test_set = set(test_cases)
    
    tv_leak = train_set.intersection(val_set)
    tt_leak = train_set.intersection(test_set)
    vt_leak = val_set.intersection(test_set)
    
    if tv_leak or tt_leak or vt_leak:
        raise ValueError(f"DATA LEAKAGE DETECTED! train-val: {tv_leak}, train-test: {tt_leak}, val-test: {vt_leak}")
        
    logger.info(f"Splits successfully created without leakage: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)}")
    
    # Save CSV files
    pd.DataFrame({"case_id": train_cases}).to_csv(os.path.join(output_dir, "train.csv"), index=False)
    pd.DataFrame({"case_id": val_cases}).to_csv(os.path.join(output_dir, "val.csv"), index=False)
    pd.DataFrame({"case_id": test_cases}).to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    return train_cases, val_cases, test_cases

def verify_splits_no_leakage(split_dir: str = "data/splits") -> bool:
    """
    Verifies that saved split CSV files exist and have zero overlap between patient sets.
    """
    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")
    test_path = os.path.join(split_dir, "test.csv")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        logger.error("One or more split CSV files are missing!")
        return False
        
    train_cases = set(pd.read_csv(train_path)["case_id"])
    val_cases = set(pd.read_csv(val_path)["case_id"])
    test_cases = set(pd.read_csv(test_path)["case_id"])
    
    assert len(train_cases.intersection(val_cases)) == 0, "Leakage between train and val"
    assert len(train_cases.intersection(test_cases)) == 0, "Leakage between train and test"
    assert len(val_cases.intersection(test_cases)) == 0, "Leakage between val and test"
    
    logger.info("Split zero-leakage assertion verified successfully!")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_dir", type=str, default="kits21/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if os.path.exists(args.cases_dir):
        cases = [d for d in os.listdir(args.cases_dir) if d.startswith("case_")]
        create_patient_splits(cases, seed=args.seed)
        verify_splits_no_leakage()
