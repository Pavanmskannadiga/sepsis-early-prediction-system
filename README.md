# 🏥 Sepsis Early Prediction System

### Explainable BiLSTM with SHAP Interpretability, SMOTE Imbalance Handling, and Cross-Hospital Generalization

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Pavanmskannadiga/sepsis-early-prediction-system)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![CI & Deployment](https://github.com/Pavanmskannadiga/sepsis-early-prediction-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Pavanmskannadiga/sepsis-early-prediction-system/actions/workflows/ci.yml)
[![Tests Passing](https://img.shields.io/badge/Tests-30%20Passed-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#)

> **Department of Artificial Intelligence & Machine Learning**  
> PES Institute of Technology & Management, Shivamogga  
> **Guide:** Prof. Shilpa K. C.  
> **Team:** Pavana M.S., Megharaja S., Srujan S. Sagar, Abubakkar Siddiq — Group 13

---

## 📋 Overview

This project implements an **end-to-end machine learning system** for predicting sepsis onset **at least 6 hours before clinical diagnosis** in ICU patients. It directly addresses three research objectives identified through a comprehensive survey of 60 papers:

| Objective | Description | Implementation |
|-----------|-------------|----------------|
| **O1** | Predict sepsis ≥6h ahead (AUC ≥ 0.90) | BiLSTM + Multi-Head Attention model |
| **O2** | Explain predictions to clinicians | SHAP DeepExplainer with doctor-readable outputs |
| **O3** | Validate across hospitals | Train on Hospital A ↔ Test on Hospital B |

## 🏗️ Architecture

```
Input (batch, 48 hours, 68 features)
    │
    │  34 clinical features + 34 missingness masks
    ↓
┌─────────────────────────────────┐
│  BiLSTM Layer 1 (128 hidden)   │  ← Bidirectional: captures forward
│  BiLSTM Layer 2 (128 hidden)   │    and backward temporal patterns
│  + LayerNorm + Dropout(0.3)    │
└─────────────┬───────────────────┘
              ↓
┌─────────────────────────────────┐
│  Multi-Head Self-Attention      │  ← 4 heads identify which hours
│  (4 heads, dim=256)             │    are most predictive
│  + Residual + LayerNorm         │
└───────┬─────────────┬───────────┘
        ↓             ↓
  Attention       Global Max
  Context          Pooling
        ↓             ↓
┌─────────────────────────────────┐
│  Concat (512) → FC(128)        │
│  + GELU + LayerNorm + Dropout   │
│  → FC(1) → Sigmoid             │
└─────────────────────────────────┘
              ↓
    P(sepsis within 6 hours)
```

## 📦 Dataset

Uses the **PhysioNet/Computing in Cardiology 2019 Challenge** dataset:
- ~40,000 ICU patients from **2 hospital systems** (A and B)
- **40 features** per hourly timestep: 8 vital signs + 26 lab values + 6 demographics
- **SepsisLabel**: Binary target with built-in 6-hour prediction window
- Natural cross-hospital split for generalization testing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/download_data.py --info
```

Follow the instructions to download from [PhysioNet](https://physionet.org/content/challenge-2019/1.0.0/) and place files in:
```
data/raw/training_setA/  (Hospital A .psv files)
data/raw/training_setB/  (Hospital B .psv files)
```

### 3. Train the Model

```bash
# Train on Hospital A with focal loss (default)
python scripts/train.py --hospital A

# Train with SMOTE oversampling
python scripts/train.py --hospital A --smote

# Train on both hospitals combined
python scripts/train.py --hospital AB

# Quick debug run
python scripts/train.py --hospital A --subset 100 --epochs 5

# Also train baseline models for comparison
python scripts/train.py --hospital A --baselines
```

### 4. Generate SHAP Explanations

```bash
# Generate SHAP analysis with trained model
python scripts/explain.py --model results/models/best_model.pt

# Use KernelExplainer (slower but more robust)
python scripts/explain.py --explainer kernel --num-samples 50
```

### 5. Cross-Hospital Validation

```bash
# Run all cross-hospital experiments
python scripts/evaluate.py --cross-hospital

# Quick evaluation
python scripts/evaluate.py --cross-hospital --subset 100 --epochs 10
```

### 6. View Results Dashboard

Open `dashboard/index.html` in your browser to see the interactive results dashboard.

## 📊 Project Structure

```
sp system/
├── config.py                     # All hyperparameters & settings
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── src/                          # Source code
│   ├── data/
│   │   ├── preprocessing.py      # PSV loading, imputation, sequences
│   │   ├── dataset.py            # PyTorch Dataset & DataLoaders
│   │   └── imbalance.py          # SMOTE, ADASYN, focal loss
│   ├── models/
│   │   ├── bilstm_attention.py   # BiLSTM + Multi-Head Attention
│   │   └── baselines.py          # LR, RF, XGBoost, LightGBM
│   ├── training/
│   │   ├── trainer.py            # Training loop, early stopping
│   │   └── metrics.py            # AUROC, AUPRC, F1, etc.
│   ├── explainability/
│   │   ├── shap_analysis.py      # SHAP computation & reporting
│   │   └── visualizations.py     # Plots & figures
│   └── evaluation/
│       ├── cross_hospital.py     # Cross-hospital experiments
│       └── results_generator.py  # Table & report generation
│
├── scripts/                      # Entry point scripts
│   ├── download_data.py          # Dataset download helper
│   ├── train.py                  # Main training script
│   ├── explain.py                # SHAP analysis script
│   └── evaluate.py               # Evaluation script
│
├── dashboard/                    # Interactive results dashboard
│   ├── index.html
│   ├── style.css
│   └── dashboard.js
│
├── data/                         # Data directory
│   ├── raw/                      # Raw PSV files
│   │   ├── training_setA/
│   │   └── training_setB/
│   └── processed/                # Preprocessed tensors
│
└── results/                      # Output directory
    ├── figures/                   # SHAP plots, ROC curves
    ├── models/                    # Saved checkpoints
    └── tables/                    # CSV performance tables
```

## 🔬 Key Features

### Imbalance Handling
- **SMOTE** oversampling with automatic reshape for 3D sequences
- **ADASYN** adaptive synthetic sampling
- **SMOTE-ENN** hybrid cleaning
- **Focal Loss** (γ=2.0, α=0.75) — preferred for temporal integrity
- **Weighted BCE** with inverse-frequency class weights
- **WeightedRandomSampler** for batch-level balancing

### Explainability (SHAP)
- **Global importance**: Top-K features ranked by mean |SHAP value|
- **Temporal analysis**: Feature importance heatmap over observation window
- **Patient-level**: Doctor-readable alerts with clinical context:
  ```
  🔴 Sepsis Risk Assessment
  Risk Score: 0.87 (HIGH RISK)
  Top Contributing Factors:
  1. ↑ Lactate: 4.8 mmol/L [ELEVATED] (SHAP: +0.185)
  2. ↓ MAP: 58 mmHg [LOW]            (SHAP: +0.152)
  3. ↑ Heart Rate: 112 bpm [ELEVATED]  (SHAP: +0.128)
  ```

### Cross-Hospital Generalization
| Experiment | AUROC | F1 |
|------------|-------|-----|
| Train A → Test A (same hospital) | 0.91 | 0.82 |
| Train A → Test B (cross-hospital) | 0.85 | 0.74 |
## 🚀 Deployment & Production Options

### 1. One-Click Cloud Deployment (Render Blueprint)
Deploy directly to Render with a single click using the pre-configured blueprint:
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Pavanmskannadiga/sepsis-early-prediction-system)

### 2. Clinical Zero-Trust Edge Deployment
Launch local live inference with an instant Cloudflare global encrypted edge tunnel:
```bash
# Windows 1-Click Launcher:
deploy_clinical_edge.bat

# Python Cross-Platform Launcher:
python deploy_clinical_edge.py
```

### 3. Containerized Deployment (Docker)
```bash
docker compose up -d --build
```

### 4. Progressive Web App (PWA) Bedside Monitoring
Open `https://<your-domain>/live.html` on any device (Chrome/Safari/Edge) and click **"Install App"** or **"Add to Home Screen"** to run full-screen with offline-first neural surrogate fallback resilience.

## 📚 References

Based on a comprehensive survey of 60 papers (1996–2026). Key references:

1. Johnson et al., "MIMIC-III database," *Scientific Data*, 2016
2. Reyna et al., "PhysioNet Challenge," *Critical Care Medicine*, 2020
3. Lundberg et al., "Explainable AI for trees," *Nature MI*, 2020
4. Hochreiter & Schmidhuber, "LSTM," *Neural Computation*, 1997

---

*Academic Year 2026 — PESITM Shivamogga, Dept. of AIML*
