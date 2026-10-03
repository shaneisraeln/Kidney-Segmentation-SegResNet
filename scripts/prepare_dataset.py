import os
import sys
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.split import create_patient_splits, verify_splits_no_leakage
from src.utils.logger import get_logger

logger = get_logger("PrepareDataset")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset patient splits")
    parser.add_argument("--data_dir", type=str, default="kits21/data")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        logger.warning(f"Data directory {args.data_dir} does not exist yet. Creating 300 case ID manifest.")
        cases = [f"case_{i:05d}" for i in range(300)]
    else:
        cases = [d for d in os.listdir(args.data_dir) if d.startswith("case_")]
        if not cases:
            cases = [f"case_{i:05d}" for i in range(300)]

    logger.info(f"Generating splits for {len(cases)} cases...")
    create_patient_splits(cases, seed=args.seed, output_dir=args.output_dir)
    verify_splits_no_leakage(args.output_dir)

if __name__ == "__main__":
    main()
