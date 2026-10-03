import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from src.inference.predict import load_trained_model, predict_ct_volume
from src.utils.logger import get_logger

logger = get_logger("InferenceCLI")

def main():
    parser = argparse.ArgumentParser(description="Run 3D Kidney & Tumor Segmentation Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input", type=str, required=True, help="Path to input CT volume (imaging.nii.gz)")
    parser.add_argument("--output", type=str, required=True, help="Path to save prediction NIfTI file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_trained_model(args.checkpoint, device=device)
    logger.info(f"Running inference on {args.input}...")

    predict_ct_volume(
        model=model,
        ct_path=args.input,
        device=device,
        output_path=args.output
    )

    logger.info(f"Inference completed! Prediction saved to {args.output}")

if __name__ == "__main__":
    main()
