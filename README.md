# Cost-Optimized Churn ML System (Production Style)

This repository implements a production-style churn decision system:
- `src/train.py` trains and saves a model
- cost-optimized threshold selection on validation set
- `src/predict.py` loads artifacts and predicts on new customer files

## Business Objective
Minimize business loss:
- False Negative (missed churn): ₹5000
- False Positive (unnecessary retention offer): ₹500

Total Cost = FP*500 + FN*5000

## Setup
```bash
pip install -r requirements.txt

## ✅ How to Run (Reproduce Results)

### 1) Create virtual environment
python -m venv .venv

### 2) Activate
# PowerShell
.\.venv\Scripts\Activate.ps1

### 3) Install dependencies
pip install -r requirements.txt

### 4) Train model + generate metrics
python src/train.py

### 5) Predict (optional)
python src/predict.py

## 📊 Final Results (Cost Optimized)

- ROC-AUC: 0.855
- PR-AUC: 0.654
- Recall: 0.98
- Precision: 0.35
- Business Cost (Test): 270000

Threshold optimized for business cost, not F1/accuracy.

## 🏗 Project Structure

cost-optimized-churn-ml-system/
│
├── data/
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│
├── models/
│   ├── model.joblib
│   ├── threshold.json
│
├── metrics/
│   ├── metrics.json
│
├── requirements.txt
└── README.md

