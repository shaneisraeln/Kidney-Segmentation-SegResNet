import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.evaluation.evaluate import evaluate_model_on_test_set
from src.utils.logger import get_logger

logger = get_logger("EvaluateCLI")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model on Test Set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best model checkpoint")
    parser.add_argument("--data_dir", type=str, default="kits21/data", help="Path to KiTS21 data directory")
    parser.add_argument("--split_dir", type=str, default="data/splits", help="Path to splits directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation results")
    args = parser.parse_args()

    logger.info(f"Starting test evaluation for checkpoint: {args.checkpoint}")
    summary = evaluate_model_on_test_set(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir
    )

    logger.info("--- TEST EVALUATION COMPLETE ---")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"{k}: {v:.4f}")
        else:
            logger.info(f"{k}: {v}")

if __name__ == "__main__":
    main()
