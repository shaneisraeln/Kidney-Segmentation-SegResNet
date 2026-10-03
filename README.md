# 3D Deep Learning Framework for Kidney and Kidney Tumor Segmentation from CT Images

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3.2-5C2D91.svg?style=flat)](https://monai.io/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-FF4B4B.svg?style=flat&logo=streamlit)](app/app.py)

An advanced, reproducible deep learning system for 3D volumetric segmentation of renal parenchyma (kidney), renal mass (tumor), and renal cyst structures from contrast-enhanced abdominal CT scans using the **KiTS21 dataset**.

---

> ⚠️ **ACADEMIC & RESEARCH DISCLAIMER**  
> This software is an independent academic research project developed strictly for scientific investigation and algorithm benchmark evaluation. **It is NOT a clinical diagnostic tool and must NOT be used for clinical medical diagnosis, treatment planning, or patient care decisions.**

---

## 🚀 Key Improvements & Innovations

1. **Lesion-Aware Multi-Class Patch Sampling (`RandCropByLabelClassesd`)**:
   - Replaced naive random background cropping with a balanced multi-class lesion sampling ratio (`10% background`, `40% kidney`, `30% tumor`, `20% cyst`).
   - Guarantees every single training batch exposes the neural network to critical tumor and cyst voxel boundaries.

2. **Class-Weighted Multi-Class Loss Function (`WeightedDiceCELoss`)**:
   - Combined Softmax Dice Loss with Class-Weighted Cross-Entropy Loss (`weights=[0.2, 1.0, 3.0, 3.0]`).
   - Prevents background dominance and forces high gradient updates on rare renal lesions.

3. **Realistic 3D Medical Image Augmentation**:
   - Implemented random 3D spatial flips, 3D rotations, zoom scaling, intensity shifts, intensity scaling, Gaussian noise, and Gaussian smoothing.

4. **Architecture Comparison**:
   - Evaluated **2D U-Net Baseline**, **Improved 3D U-Net**, and **MONAI SegResNet** on the identical untouched patient test split.

---

## 📊 Final Benchmark Results Matrix

### Validation Performance (Model Selection Criteria)

| Model Architecture | Val Mean Dice | Kidney Dice | Tumor Dice | Cyst Dice | Model Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2D U-Net Baseline** | `0.0164` | `0.0370` | `0.0095` | `0.0014` | Baseline Reference |
| **Improved 3D U-Net** | `0.2498` | `0.4455` | `0.2677` | `0.0362` | Proposed 3D Model |
| **MONAI SegResNet** | **`0.6051`** | **`0.9436`** | **`0.8716`** | `0.0000` | **Best Selected Model** |

---

### Untouched Held-Out Test Set Results (`results/summary_metrics.json`)

Evaluated strictly on the held-out test split using sliding-window inference:

| Metric | Score / Value | Description |
| :--- | :---: | :--- |
| **Kidney Parenchyma Dice** | **`0.9420`** | **94.20% Organ Segmentation Accuracy** |
| **Kidney Parenchyma IoU** | **`0.8904`** | **89.04% Intersection over Union** |
| **Kidney Mass / Tumor Dice** | **`0.8714`** | **87.14% Lesion Segmentation Accuracy** |
| **Kidney Mass / Tumor IoU** | **`0.7720`** | **77.20% Intersection over Union** |
| **Tumor Boundary HD95** | **`2.00 mm`** | **2.0 mm 95th Percentile Hausdorff Distance** |
| **Overall Mean Dice** | **`0.6045`** | **Macro Average across all classes** |
| **KiTS Composite Kidney Dice** | **`0.8641`** | **Organ + Mass Combined Region** |

---

## 📦 Directory Structure

```text
Kidney Recognition/
├── PRD.md                       # Product Requirements Document
├── TRD.md                       # Technical Requirements Document
├── README.md                    # Project Documentation
├── requirements.txt             # Environment Dependencies
├── kits21/                      # KiTS21 Repository & Imaging Data
├── configs/                     # YAML Configuration Files
│   ├── baseline_2d.yaml
│   ├── unet3d.yaml
│   └── segresnet.yaml
├── data/
│   └── splits/                  # Patient-Level CSV Splits (Train/Val/Test)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── utils/                   # Config, Logger, Seed Utilities
│   ├── data/                    # Audit, Preprocessing, Label Mapping, Dataset
│   ├── models/                  # UNet2D, UNet3D, SegResNet3D Architectures
│   ├── training/                # Weighted DiceCE Loss, Metrics, Trainer Engine
│   ├── inference/               # Sliding-Window Volumetric Predictor
│   ├── evaluation/              # Test Set Evaluation & Error Analysis
│   └── visualization/           # Triplane & Plotting Utilities
├── scripts/                     # Executable Scripts
│   ├── download_kits.py
│   ├── audit_dataset.py
│   ├── prepare_dataset.py
│   └── generate_plots.py
├── checkpoints/                 # Model Checkpoints (.pth)
├── experiments/                 # Experiment Registry CSV
├── results/                     # Metric CSVs, JSON Summaries, Error Cases
├── tests/                       # Automated PyTest Suite
└── app/                         # Streamlit Interactive Web Application
    └── app.py
```

---

## ⚡ Quick Start & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train Best Model (MONAI SegResNet)
python train.py --config configs/segresnet.yaml

# 3. Evaluate on Untouched Test Set
python evaluate.py --checkpoint checkpoints/segresnet/best_model.pth

# 4. Single CT Scan Volumetric Inference
python inference.py \
    --checkpoint checkpoints/segresnet/best_model.pth \
    --input kits21/data/case_00000/imaging.nii.gz \
    --output predictions/case_00000_pred.nii.gz

# 5. Launch Streamlit Interactive Workspace
streamlit run app/app.py

# 6. Run PyTest Unit Tests
python -m pytest tests/
```

---

## 📜 Citation & License

- **KiTS21 Dataset:** Heller et al., *"The KiTS21 Challenge Dataset"*, https://kits-challenge.org/kits21/
- **License:** Apache 2.0
