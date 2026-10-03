# TRD — 3D Deep Learning Framework for Kidney and Kidney Tumor Segmentation

## 1. Technical Overview

This document defines the technical implementation for an advanced machine-learning system for volumetric kidney and kidney-tumor segmentation using the KiTS21 dataset.

The system consists of:

```text
Raw KiTS21
    ↓
Dataset Audit
    ↓
Preprocessing
    ↓
2D Baseline / 3D U-Net
    ↓
Training
    ↓
Inference
    ↓
Post-processing
    ↓
Evaluation
    ↓
Visualization / Demo
```

---

## 2. Technology Stack

### Language

Python 3.10+ or the most compatible supported version for the selected PyTorch/MONAI environment.

### Core ML

- PyTorch
- MONAI

### Medical Imaging

- NiBabel
- SimpleITK where required
- MONAI transforms

### Numerical

- NumPy
- SciPy

### Visualization

- Matplotlib
- Plotly where useful
- Optional ITK/VTK-based visualization if required

### Data Analysis

- Pandas

### Application

- Streamlit

### Experiment Tracking

Preferred:

- MLflow

Acceptable lightweight alternative:

- JSON/CSV experiment logging

---

## 3. Dataset Acquisition

Use the official KiTS21 repository:

https://github.com/neheller/kits21

The official repository provides segmentation data and `get_imaging` scripts for downloading the imaging data into the local `data/` directory.

Use the official Python downloader unless there is a technical reason to use another supported method.

Expected structure:

```text
kits21/
├── data/
│   ├── case_00000/
│   │   ├── imaging.nii.gz
│   │   ├── aggregated_OR_seg.nii.gz
│   │   ├── aggregated_AND_seg.nii.gz
│   │   ├── aggregated_MAJ_seg.nii.gz
│   │   └── segmentations/
│   ├── case_00001/
│   └── ...
└── kits.json
```

The official KiTS21 repository states that the training set contains cases `case_00000` through `case_00299`.

Do not use an unofficial mirror unless the official download mechanism fails and the alternative is explicitly documented.

---

## 4. Dataset Audit Module

Create:

```text
src/data/audit.py
```

The audit must inspect all 300 cases and produce:

```text
results/dataset_audit.json
results/dataset_summary.csv
```

Collect:

- Case ID
- Image shape
- Image dtype
- Voxel spacing
- Orientation
- Minimum HU
- Maximum HU
- Mean HU
- Standard deviation
- Label values
- Label voxel counts
- Missing files
- Corrupt files
- Kidney presence
- Tumor presence
- Cyst presence

Generate summary statistics and plots.

---

## 5. Label Verification

Before training, inspect the official KiTS21 label representation.

Do not assume that `1`, `2`, etc. mean specific structures without verification.

The pipeline must establish and document the final training label mapping.

If the official aggregated segmentation is selected, document which aggregation is used and why.

Preferred default:

```text
aggregated_MAJ_seg.nii.gz
```

if it produces the desired semantic classes after verification.

Do not silently alter the original labels.

---

## 6. Dataset Split

Create:

```text
data/splits/
├── train.csv
├── val.csv
└── test.csv
```

Use deterministic patient-level splitting.

Recommended:

```text
70% train
15% validation
15% test
```

Use a fixed seed.

Before training, run a leakage check:

```text
intersection(train, val) == empty
intersection(train, test) == empty
intersection(val, test) == empty
```

Fail loudly if leakage is detected.

---

## 7. Preprocessing Pipeline

Create:

```text
src/data/preprocessing.py
```

Implement configurable preprocessing.

### Stage 1 — Load

Read NIfTI using MONAI/NiBabel.

### Stage 2 — Orientation

Normalize orientation consistently.

### Stage 3 — Spacing

Resample image and segmentation to a configurable target spacing.

Use:

- Linear/trilinear interpolation for image
- Nearest-neighbor interpolation for segmentation

Never use linear interpolation for categorical labels.

### Stage 4 — Intensity

Inspect KiTS HU distribution.

Use configurable clipping/windowing.

Do not copy external paper parameters without dataset evidence.

### Stage 5 — Normalization

Normalize the processed image using a documented strategy.

### Stage 6 — Foreground/ROI

Where appropriate, crop irrelevant empty background.

### Stage 7 — Patch Extraction

Use 3D patches suitable for available GPU memory.

Initial candidates:

```text
96 × 96 × 96
128 × 128 × 128
```

Benchmark memory before committing.

---

## 8. Augmentation

Training-only augmentations:

- Random spatial crop
- Random flip
- Small random rotation
- Random intensity shift
- Random intensity scaling
- Gaussian noise where justified

Do not augment validation/test data except for deterministic preprocessing.

All augmentations must preserve segmentation-label alignment.

---

## 9. Baseline Architecture

Implement:

```text
src/models/unet2d.py
```

A conventional 2D U-Net baseline.

Because the data is volumetric, the baseline should operate slice-wise.

Pipeline:

```text
3D CT
 ↓
2D slices
 ↓
2D U-Net
 ↓
slice predictions
 ↓
reconstruct 3D mask
```

Evaluate the reconstructed 3D prediction using the same test metrics as the 3D model.

---

## 10. Proposed Architecture

Implement:

```text
src/models/unet3d.py
```

Use a 3D U-Net.

Initial design:

- 3D convolution
- Encoder-decoder
- Skip connections
- Configurable feature channels
- Multi-class output
- Softmax during inference

Use MONAI where possible rather than unnecessarily implementing established components from scratch.

---

## 11. Optional Advanced Model

Only after the 3D U-Net pipeline is stable.

Possible choice:

```text
SegResNet
```

or:

```text
Residual 3D U-Net
```

The final project should not become an architecture zoo.

One meaningful advanced extension is sufficient.

---

## 12. Loss

Implement:

```text
src/training/losses.py
```

Initial loss:

```text
DiceLoss
```

Optional comparison:

```text
DiceLoss + CrossEntropyLoss
```

Evaluate whether the combined loss improves validation performance.

Loss configuration must be stored in the experiment configuration.

---

## 13. Optimizer

Initial:

```text
AdamW
```

Recommended starting configuration:

```yaml
learning_rate: 0.0003
weight_decay: 0.00001
```

These are starting values, not guaranteed optimal values.

Allow configuration overrides.

---

## 14. Scheduler

Implement a configurable scheduler.

Preferred initial approach:

```text
CosineAnnealingLR
```

or:

```text
ReduceLROnPlateau
```

The agent may select the better option based on observed training behavior, but must document the choice.

---

## 15. Training Engine

Create:

```text
src/training/trainer.py
train.py
```

Training engine must support:

- AMP/mixed precision
- Gradient scaling
- Gradient accumulation
- Checkpointing
- Resume training
- Early stopping
- Learning-rate scheduling
- TensorBoard/MLflow logging
- Periodic validation

Checkpoint:

```text
checkpoints/
├── best_model.pth
└── last_model.pth
```

Save:

- model state
- optimizer state
- scheduler state
- epoch
- best metric
- configuration
- random seed

---

## 16. Memory Safety

3D models are memory intensive.

Implement:

- Configurable patch size
- Configurable batch size
- AMP
- Gradient accumulation
- GPU memory logging

If OOM occurs:

1. Reduce batch size.
2. Enable/verify AMP.
3. Reduce patch size.
4. Use gradient accumulation.
5. Only then reduce model width.

Never silently skip training samples because of GPU memory issues.

---

## 17. Inference

Create:

```text
src/inference/predict.py
```

Use sliding-window inference for full CT volumes.

Requirements:

- Overlapping patches
- Configurable overlap
- Reconstruct full-volume probability map
- Argmax to segmentation
- Preserve original spatial metadata
- Save prediction as NIfTI

Output:

```text
predictions/
└── case_XXXXX_pred.nii.gz
```

---

## 18. Post-processing

Implement configurable post-processing.

Potential operations:

- Remove isolated components
- Keep anatomically relevant connected components
- Optional small-component filtering

Do not apply aggressive post-processing without evaluating its effect on validation metrics.

---

## 19. Evaluation Engine

Create:

```text
src/evaluation/metrics.py
evaluate.py
```

Calculate:

### Per-class

- Dice
- IoU
- HD95

### Aggregate

- Macro mean
- Median
- Standard deviation

Also calculate case-level metrics.

Output:

```text
results/
├── test_metrics.csv
├── summary_metrics.json
└── model_comparison.csv
```

---

## 20. Evaluation Protocol

The test set must be completely held out.

No:

- Hyperparameter tuning on test
- Architecture selection using test
- Threshold tuning on test
- Repeated test-driven iteration

Validation set is used for model selection.

Test set is used once for final reported performance, with additional analysis only after the final model is frozen.

---

## 21. Error Analysis

Create:

```text
src/evaluation/error_analysis.py
```

Sort test cases by:

- Mean Dice
- Tumor Dice
- Kidney Dice

Generate:

```text
results/error_cases/
├── best_01.png
├── best_02.png
├── worst_01.png
├── worst_02.png
└── ...
```

Each visualization should contain:

```text
Original CT
Ground Truth
Prediction
Overlay
```

Where possible, calculate tumor-volume bins and investigate whether segmentation performance changes with lesion size.

---

## 22. Experiment Configuration

Use:

```text
configs/
├── baseline_2d.yaml
├── unet3d.yaml
└── optional_model.yaml
```

Example:

```yaml
seed: 42

dataset:
  name: KiTS21
  split_file: data/splits/train.csv

model:
  name: unet3d
  in_channels: 1
  out_channels: <verified_class_count>

training:
  epochs: 200
  batch_size: 2
  patch_size: [96, 96, 96]
  learning_rate: 0.0003
  weight_decay: 0.00001
  early_stopping_patience: 30
  amp: true

loss:
  name: dice

evaluation:
  metrics:
    - dice
    - iou
    - hd95
```

The exact class count and configuration must be populated after dataset audit.

---

## 23. Experiment Registry

Maintain:

```text
experiments/experiment_log.csv
```

Columns:

```text
experiment_id
timestamp
dataset_version
split_seed
model
loss
optimizer
learning_rate
batch_size
patch_size
epochs
best_epoch
best_val_dice
test_dice
test_iou
test_hd95
training_time
gpu
notes
```

---

## 24. Visualization Module

Create:

```text
src/visualization/
```

Functions for:

- Axial slice visualization
- Coronal visualization
- Sagittal visualization
- Ground-truth overlay
- Prediction overlay
- Error map
- Training curves
- Model comparison charts
- 3D surface/volume rendering where feasible

Save publication-quality figures to:

```text
results/figures/
```

---

## 25. Streamlit Application

Create:

```text
app/app.py
```

Features:

### Input

Upload/select a NIfTI CT volume.

### Processing

Run trained 3D U-Net inference.

### Output

Display:

- CT slices
- Predicted segmentation overlay
- Class legend
- 3D visualization if practical
- Predicted class volumes if implemented

If ground truth is available in demo mode, optionally display Dice/IoU.

The application must clearly display:

> Research prototype. Not intended for clinical diagnosis or treatment decisions.

---

## 26. Testing

Create automated tests for:

### Data

- NIfTI loading
- Label validation
- Shape matching
- Spacing handling

### Preprocessing

- Image/label alignment
- Normalization
- Resampling
- Patch generation

### Model

- Forward pass
- Output dimensions
- Multi-class output

### Evaluation

- Dice implementation
- IoU implementation
- Edge cases

### Inference

- Sliding-window reconstruction
- NIfTI output integrity

---

## 27. Performance Requirements

The system should:

- Use GPU when available.
- Fall back to CPU for preprocessing/inference if necessary.
- Avoid loading the complete dataset into RAM.
- Stream/load cases as needed.
- Cache processed data where beneficial.
- Log processing and training times.

---

## 28. Security / Safety

No patient-identifying information should be exposed in the application.

Do not send CT data to external APIs.

All inference should be local.

Do not describe outputs as medical diagnoses.

---

## 29. Reproducibility

README must provide commands similar to:

```bash
# Environment
pip install -r requirements.txt

# Dataset
python scripts/download_kits.py

# Audit
python scripts/audit_dataset.py

# Prepare
python scripts/prepare_dataset.py

# Train baseline
python train.py --config configs/baseline_2d.yaml

# Train proposed model
python train.py --config configs/unet3d.yaml

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pth

# Inference
python inference.py --input <path> --output <path>

# App
streamlit run app/app.py
```

The actual commands may differ, but all functionality must be documented.

---

## 30. Definition of Done

Technical implementation is complete when:

- [ ] Official KiTS21 repository cloned.
- [ ] All 300 cases downloaded.
- [ ] Dataset integrity verified.
- [ ] Label semantics verified.
- [ ] EDA generated.
- [ ] Patient-level split generated.
- [ ] Preprocessing verified visually.
- [ ] 2D U-Net baseline implemented.
- [ ] 2D baseline trained.
- [ ] 3D U-Net implemented.
- [ ] 3D model trained.
- [ ] Validation monitoring works.
- [ ] Best checkpoint saved.
- [ ] Final test evaluation works.
- [ ] Dice/IoU/HD95 calculated.
- [ ] 2D vs 3D comparison generated.
- [ ] Error analysis generated.
- [ ] Visualization pipeline works.
- [ ] Inference on unseen CT works.
- [ ] Streamlit demo works.
- [ ] README is complete.
- [ ] No fabricated results or undocumented experiments exist.
