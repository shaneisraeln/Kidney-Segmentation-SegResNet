import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.audit import run_dataset_audit

def main():
    parser = argparse.ArgumentParser(description="Audit KiTS21 Dataset")
    parser.add_argument("--data_dir", type=str, default="kits21/data")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    run_dataset_audit(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()
