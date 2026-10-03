import os
import json
import pandas as pd
import torch
import numpy as np
from typing import Dict, Any, List
from monai.inferers import sliding_window_inference
from src.utils.logger import get_logger
from src.training.metrics import SegmentationMetrics
from src.inference.predict import load_trained_model
from src.data.dataset import create_dataloaders

from src.utils.seed import seed_everything

logger = get_logger("Evaluation")

def evaluate_model_on_test_set(
    checkpoint_path: str,
    data_dir: str = "kits21/data",
    split_dir: str = "data/splits",
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Evaluates trained model checkpoint on the held-out test split.
    Saves test_metrics.csv and summary_metrics.json.
    """
    seed_everything(42)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_trained_model(checkpoint_path, device=device)
    _, _, test_loader = create_dataloaders(data_dir=data_dir, split_dir=split_dir, batch_size=1)

    metrics_engine = SegmentationMetrics(num_classes=4)
    case_results = []

    logger.info(f"Evaluating {len(test_loader)} test cases...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_id = batch.get("case_id", ["unknown"])[0]

            logits = sliding_window_inference(
                inputs=images,
                roi_size=(48, 64, 64),
                sw_batch_size=2,
                predictor=model,
                overlap=0.5
            )

            # Compute case-level metrics
            single_metric = SegmentationMetrics(num_classes=4)
            single_metric.update(logits, labels)
            res = single_metric.compute()
            res["case_id"] = case_id
            case_results.append(res)

            metrics_engine.update(logits, labels)

    summary_metrics = metrics_engine.compute()
    
    # Save case-level test metrics CSV
    df_cases = pd.DataFrame(case_results)
    csv_path = os.path.join(output_dir, "test_metrics.csv")
    df_cases.to_csv(csv_path, index=False)
    logger.info(f"Saved test metrics per case to {csv_path}")

    # Save summary JSON
    json_path = os.path.join(output_dir, "summary_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"Saved overall test summary metrics to {json_path}")

    return summary_metrics
