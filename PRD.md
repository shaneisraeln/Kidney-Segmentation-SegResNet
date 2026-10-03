# PRD — 3D Deep Learning Framework for Kidney and Kidney Tumor Segmentation

## 1. Product Overview

### Project Name
3D Deep Learning Framework for Kidney and Kidney Tumor Segmentation from CT Images

### Project Type
Advanced Machine Learning / Medical Image Segmentation

### Primary Objective
Build a reproducible machine-learning system that takes a contrast-enhanced abdominal CT volume from the KiTS21 dataset and automatically produces voxel-level segmentation of renal structures, with emphasis on kidney and kidney-tumor segmentation.

The project is an independent implementation using KiTS21. It must NOT claim to reproduce, replicate, or use results from any previously studied research paper.

---

## 2. Problem Statement

Manual delineation of kidneys and renal tumors in 3D CT images is labor-intensive and requires expert annotation. The objective is to investigate whether a 3D deep-learning segmentation model can learn volumetric anatomical context and produce accurate kidney/tumor masks.

### Research Question

Can a 3D deep-learning segmentation model provide better volumetric kidney and tumor segmentation than a simpler 2D segmentation baseline when trained and evaluated on the KiTS21 dataset?

---

## 3. Target Users

This is an academic/research prototype intended for:

- Students and researchers studying medical image segmentation
- Demonstration of advanced machine learning techniques
- Experimental evaluation of 2D vs 3D segmentation

This is NOT a clinical diagnostic product.

---

## 4. Dataset

### Primary Dataset
KiTS21 — Kidney and Kidney Tumor Segmentation Challenge.

Official repository:
https://github.com/neheller/kits21

Official challenge:
https://kits-challenge.org/kits21/

The KiTS21 training set contains 300 CT cases. The dataset provides expert annotations for kidney, kidney tumor, and kidney cyst regions. The official repository stores segmentation data and provides scripts for downloading the imaging data.

### Dataset Policy

- Download the complete available KiTS21 training dataset.
- Preserve raw data unchanged.
- Never silently discard cases.
- Maintain a manifest of all cases.
- Record failed/corrupt/missing cases explicitly.
- Use only publicly available data.
- Follow the dataset license and cite KiTS21 appropriately.

---

## 5. ML Task

The primary task is 3D semantic segmentation.

### Input

A volumetric contrast-enhanced CT scan.

### Output

A voxel-level segmentation mask containing the available KiTS21 target classes.

The implementation must first inspect the actual label encoding and verify the meaning of every label before training.

The agent must not assume class IDs without inspecting the downloaded dataset metadata/labels.

### Recommended Initial Target

Multi-class segmentation:

- Background
- Kidney
- Tumor
- Cyst

If the downloaded KiTS21 annotation representation requires a different class construction, document the exact mapping and preserve the official semantics.

---

## 6. Core User Flow

```text
CT Volume
    ↓
Dataset Validation
    ↓
Preprocessing
    ↓
3D Patch Extraction
    ↓
3D Segmentation Model
    ↓
Predicted Segmentation
    ↓
Post-processing
    ↓
Evaluation
    ↓
Visualization
```

The final prototype should additionally support:

```text
CT Volume
    ↓
Inference
    ↓
Segmentation Overlay
    ↓
3D Visualization
```

---

## 7. Baseline and Proposed Models

### Baseline

Implement a 2D U-Net baseline.

Purpose:

- Establish a meaningful reference point.
- Evaluate segmentation without explicit volumetric context.

### Proposed Advanced Model

Implement a 3D U-Net.

Purpose:

- Learn 3D spatial/anatomical context.
- Exploit inter-slice information.
- Provide a direct advanced-ML comparison against the 2D baseline.

### Optional Extension

Only after the baseline and 3D U-Net are working, optionally evaluate one improvement such as:

- Residual 3D U-Net
- Attention U-Net
- MONAI SegResNet

Do not add multiple architectures merely for complexity.

---

## 8. Success Criteria

The project is successful when all of the following are satisfied:

### Dataset

- All available 300 KiTS21 training cases are downloaded.
- Dataset audit completes successfully.
- Label mappings are verified.
- No patient leakage exists between splits.

### ML

- 2D baseline trains successfully.
- 3D U-Net trains successfully.
- Best checkpoints are saved.
- Validation performance is monitored.
- Final evaluation is performed only on the held-out test set.

### Evaluation

The system reports at minimum:

- Dice Similarity Coefficient
- Intersection over Union
- HD95 or another boundary metric where computationally practical
- Per-class metrics
- Mean/median/std where appropriate

### Analysis

- 2D vs 3D comparison is provided.
- Worst-performing cases are identified.
- Visual error analysis is performed.
- Failure categories are based on actual observations rather than assumptions.

### Application

A working local inference/demo interface is available.

---

## 9. Data Split

Use patient-level splitting.

Suggested initial split:

- 70% training
- 15% validation
- 15% test

The exact split may be changed if justified by dataset characteristics, but:

1. Split must be reproducible.
2. Test cases must be fixed before model development.
3. Test cases must not be used for hyperparameter tuning.
4. Case IDs must be saved in train.csv, val.csv, and test.csv.

---

## 10. Preprocessing Requirements

The agent must inspect the dataset before selecting preprocessing parameters.

Potential pipeline:

```text
NIfTI CT
    ↓
Orientation normalization
    ↓
Voxel spacing normalization
    ↓
Intensity clipping / normalization
    ↓
Foreground or body-region cropping
    ↓
Patch sampling
    ↓
Augmentation
```

Do NOT blindly copy preprocessing parameters from external papers.

All preprocessing choices must be documented with rationale.

---

## 11. Training Requirements

The training system should support:

- Mixed precision where safe
- Configurable batch size
- Configurable patch size
- Configurable learning rate
- Adam or AdamW
- Dice-based loss
- Optional Dice + Cross Entropy loss
- Learning-rate scheduling
- Early stopping
- Model checkpointing
- Resume-from-checkpoint
- Seed control

The system must automatically adapt to available GPU memory rather than crashing because of an oversized 3D patch.

---

## 12. Experimentation

Maintain a structured experiment log.

Minimum experiments:

### E01 — 2D U-Net baseline

### E02 — 3D U-Net

### E03 — Optional improvement

Each experiment must record:

- Dataset split
- Model
- Loss
- Optimizer
- Learning rate
- Batch size
- Patch size
- Epochs
- Best validation metric
- Final test metrics
- Training time
- GPU/environment
- Notes

Never fabricate missing experiment results.

---

## 13. Evaluation

### Primary metric

Dice Similarity Coefficient.

### Secondary metrics

- IoU
- HD95
- Per-class Dice
- Per-class IoU

If a metric cannot be computed reliably, document why rather than substituting an invented value.

### Required comparison

2D U-Net vs 3D U-Net.

Example:

| Model | Kidney Dice | Tumor Dice | Cyst Dice | Mean Dice |
|---|---:|---:|---:|---:|
| 2D U-Net | Actual | Actual | Actual | Actual |
| 3D U-Net | Actual | Actual | Actual | Actual |

Do not populate this table with expected or published values.

---

## 14. Error Analysis

Perform actual visual and quantitative analysis.

Identify:

- Best cases
- Worst cases
- False positives
- False negatives
- Under-segmentation
- Over-segmentation
- Boundary errors

Where possible, investigate performance against:

- Tumor size
- Tumor volume
- Anatomical location
- Image quality
- Case complexity

Only report relationships that are actually measured.

Generate overlays:

- CT
- Ground truth
- Prediction
- Error/difference

---

## 15. Visualization Requirements

Generate:

1. Dataset sample CT slices.
2. Ground-truth overlays.
3. Prediction overlays.
4. Axial/coronal/sagittal views.
5. 3D segmentation visualization where practical.
6. Training/validation loss curves.
7. Dice curves.
8. Model comparison plots.
9. Error-case visualizations.

---

## 16. Application Requirements

Build a lightweight local demo, preferably Streamlit.

Minimum functionality:

1. Load/select a CT volume.
2. Run inference using the trained model.
3. Display CT slices.
4. Overlay predicted segmentation.
5. Display class information.
6. Display basic segmentation metrics when ground truth is available.
7. Provide a 3D visualization if practical.

The UI must clearly state:

> Research prototype — not for clinical diagnosis.

---

## 17. Non-Functional Requirements

### Reproducibility

The project must include:

- requirements.txt or environment.yml
- configuration file
- fixed random seeds
- dataset manifest
- split files
- training commands
- evaluation commands
- inference commands

### Reliability

The pipeline must:

- Detect missing files.
- Detect invalid NIfTI volumes.
- Validate label values.
- Avoid patient leakage.
- Save checkpoints.
- Resume interrupted training.

### Maintainability

Use modular Python code rather than one giant notebook.

---

## 18. Deliverables

The agent must produce:

### Code

- Dataset preparation
- EDA
- Preprocessing
- 2D U-Net
- 3D U-Net
- Training
- Evaluation
- Error analysis
- Visualization
- Inference
- Demo application

### Artifacts

- Trained model checkpoints
- Experiment logs
- Metric CSV/JSON files
- Figures
- Tables
- Dataset split manifests

### Documentation

- README
- Installation guide
- Dataset preparation guide
- Training guide
- Evaluation guide
- Inference guide
- Architecture explanation
- Experiment summary

---

## 19. Academic Integrity Rules

The implementation must never:

- Fabricate metrics.
- Copy published results as project results.
- Claim an experiment was performed when it was not.
- Hide failed experiments.
- Modify the test set to improve results.
- Claim clinical validity.
- Claim medical diagnostic capability.

All final numerical results must originate from actual project runs.

---

## 20. Definition of Done

The project is considered complete when:

- [ ] KiTS21 dataset is downloaded and verified.
- [ ] Dataset EDA is complete.
- [ ] Patient-level split is created.
- [ ] Preprocessing pipeline is validated visually.
- [ ] 2D U-Net baseline is trained.
- [ ] 3D U-Net is trained.
- [ ] Final test evaluation is complete.
- [ ] 2D vs 3D comparison is available.
- [ ] Error analysis is complete.
- [ ] Visualizations are generated.
- [ ] Inference pipeline works on unseen test data.
- [ ] Demo application works locally.
- [ ] README allows another person to reproduce the workflow.
- [ ] No fabricated results exist.
