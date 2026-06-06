# TFG Automated Skin Lesion Analysis with Deep Learning and LLM Integration

**Author:** Natalia García Sánchez  
**Supervisor:** Rubén Juárez Cádiz  
**Institution:** CEU San Pablo University, Madrid  
**Degree:** Biomedical Engineering  
**Year:** 2026

---

## Overview

This repository contains the full source code for a TFG (Bachelor's Thesis) on automated dermoscopic skin lesion analysis. The system combines three components:

1. **Lesion segmentation** — U-Net++ with EfficientNet-B5 encoder trained on ISIC 2018
2. **Eight-class classification** — EfficientNet-B4 trained on ISIC 2019, with a controlled comparison between classification on original images (Model A) and segmentation-masked images (Model B)
3. **LLM-based explanation generation** — conversational diagnostic prototype connecting the trained models to Claude via the Model Context Protocol (MCP)

---

## Repository Structure

```
tfg-skin-lesion/
│
├── segmentation_training.py       # Train U-Net++ segmentation model
├── segmentation_evaluation.py     # Evaluate segmentation model + qualitative analysis
│
├── classification_training.py     # Train EfficientNet-B4 classification models (A and B)
├── classification_evaluation.py   # Evaluate models + TTA + McNemar + threshold tuning
│
├── models.py                      # MCP server model loader (predict function)
├── server.py                      # MCP server exposing analyze_skin_lesion tool
│
└── README.md
```
---

## Datasets

This project uses two publicly available ISIC datasets. They are **not included** in this repository and must be downloaded separately.

**ISIC 2018 — Segmentation**
- Training images + masks: https://challenge.isic-archive.com/landing/2018/
- Official test set: same link above
- Expected paths:
  - `/workspace/datasets/ISIC2018/images/ISIC2018_Task1-2_Training_Input/`
  - `/workspace/datasets/ISIC2018/masks/ISIC2018_Task1_Training_GroundTruth/`
  - `/workspace/datasets/ISIC2018/images/ISIC2018_Task1-2_Test_Input/`
  - `/workspace/datasets/ISIC2018/masks/ISIC2018_Task1_Test_GroundTruth/`

**ISIC 2019 — Classification**
- Training images + labels: https://challenge.isic-archive.com/landing/2019/
- Official test set + labels: same link above
- Expected paths:
  - `/workspace/datasets/ISIC2019/images/ISIC_2019_Training_Input/`
  - `/workspace/datasets/ISIC2019/images/ISIC_2019_Test_Input/`
  - `/workspace/datasets/ISIC2019/labels/ISIC_2019_Training_GroundTruth.csv`
  - `/workspace/datasets/ISIC2019/labels/ISIC_2019_Test_GroundTruth.csv`

---

## Trained Models

The trained model weights are **not included** in this repository due to file size. They are available on request.

| File | Description |
|---|---|
| `unet_best.pth` | Best U-Net++ checkpoint (ISIC 2018) |
| `efficientnet_A_original.pth` | Best EfficientNet-B4 Model A checkpoint |
| `efficientnet_B_masked.pth` | Best EfficientNet-B4 Model B checkpoint |

---

## Requirements

```bash
pip install torch torchvision segmentation-models-pytorch \
            scikit-learn pandas matplotlib pillow statsmodels
```

Trained on a RunPod cloud instance with NVIDIA RTX A5000 (24 GB VRAM).

---

## How to Run

### 1. Segmentation

```bash
# Train
python segmentation_training.py

# Evaluate
python segmentation_evaluation.py
```

### 2. Classification

```bash
# Train (requires unet_best.pth in /workspace/)
python classification_training.py

# Evaluate (requires trained .pth models)
python classification_evaluation.py
```

### 3. MCP Prototype

Requires Claude Desktop with MCP support. Add the following to your Claude Desktop config file:

```json
{
  "mcpServers": {
    "skin-lesion": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

Then place a dermoscopic image path in Claude Desktop and ask for an analysis.

---

## Results

### Segmentation — Official ISIC 2018 Test Set (n=1,000)

| Metric | U-Net baseline | U-Net++ (final) |
|---|---|---|
| Dice (mean) | 0.873 | **0.885** |
| IoU | 0.800 | **0.806** |
| Precision | 0.897 | 0.845 |
| Recall | 0.893 | **0.955** |
| Specificity | 0.971 | 0.927 |

### Classification — Official ISIC 2019 Test Set (n=6,191)

| Metric | Model A base | Model A + TTA | Model A + TTA + Thresh |
|---|---|---|---|
| Accuracy | 46.63% | 48.15% | **59.38%** |
| Balanced Accuracy | 55.69% | 57.36% | **59.31%** |
| Macro F1 | 0.395 | 0.410 | **0.469** |
| Macro AUC-ROC | 0.887 | 0.892 | — |

Model A consistently outperformed Model B (segmentation-masked) across all metrics.  
McNemar test: χ²=9.28, p=0.002 — difference is statistically significant.

---

## License

This project is released for academic purposes. The ISIC datasets are subject to their own terms of use available at https://www.isic-archive.com.
