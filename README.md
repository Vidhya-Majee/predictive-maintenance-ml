<div align="center">

# 🔧 Predictive Maintenance of Industrial Machinery

**ML-powered multi-class failure detection using real-world sensor data**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-189AC2?style=flat-square)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=flat-square)](https://shap.readthedocs.io/)
[![IBM Watson Studio](https://img.shields.io/badge/IBM%20Watson%20Studio-Cloud%20Lite-1261FE?style=flat-square&logo=ibm&logoColor=white)](https://www.ibm.com/cloud/watson-studio)

> **AICTE-2026 Problem Statement No. 39** — Mechanical Engineering ML Track

</div>

---

## 📌 Problem Statement

Unexpected machine failures in industrial environments cause costly downtime and safety risks. The goal of this project is to build a machine learning pipeline that **predicts the type of failure** before it occurs — enabling proactive maintenance scheduling and reducing unplanned outages.

Given sensor readings from manufacturing equipment, the model classifies each observation into one of **6 failure categories**:

| Failure Type | Count | % of Dataset |
|---|---|---|
| ✅ No Failure | 9,652 | 96.5% |
| 🌡️ Heat Dissipation Failure | 112 | 1.1% |
| ⚡ Power Failure | 95 | 0.95% |
| 💪 Overstrain Failure | 78 | 0.78% |
| 🔩 Tool Wear Failure | 45 | 0.45% |
| 🎲 Random Failures | 18 | 0.18% |

The **severe class imbalance** (96.5% non-failure) is the central challenge of this problem.

---

## 📊 Dataset

**Source:** [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) on Kaggle

| Property | Value |
|---|---|
| Rows | 10,000 |
| Features | 7 (after preprocessing) |
| Target | `Failure Type` (6 classes) |
| Missing Values | None |

### Features

| Feature | Type | Description |
|---|---|---|
| `Air temperature [K]` | float | Ambient air temperature |
| `Process temperature [K]` | float | Machine process temperature |
| `Rotational speed [rpm]` | int | Spindle rotation speed |
| `Torque [Nm]` | float | Applied torque |
| `Tool wear [min]` | int | Cumulative tool wear time |
| `Type_L` / `Type_M` | bool | Product quality tier (Low / Medium / High) |

---

## 🧠 ML Approach

### Pipeline Overview

```
Raw Data (10,000 × 10)
       │
       ▼
  Preprocessing
  ├─ Drop UDI, Product ID
  ├─ One-hot encode Type (L/M/H)
  └─ Stratified 80/20 train-test split
       │
       ▼
  Model Training
  ├─ Baseline: Decision Tree (max_depth=3)
  ├─ Random Forest (200 estimators, class_weight='balanced')
  └─ XGBoost ← best model
       │
       ▼
  Explainability
  └─ SHAP TreeExplainer (class-level feature importance)
```

### Models Trained

#### 1. 🌿 Decision Tree (Baseline)
- `max_depth=3`, `random_state=42`
- Establishes a naive upper-bound on accuracy due to class imbalance

#### 2. 🌲 Random Forest
- `n_estimators=200`, `class_weight='balanced'`, `random_state=42`
- Balanced class weights to penalize misclassification of minority failure types

#### 3. ⚡ XGBoost *(Best Model)*
- `eval_metric='mlogloss'`, LabelEncoder for target encoding
- Column names sanitized (removed `[` `]`) for XGBoost compatibility
- Strongest performance across minority failure classes

---

## 📈 Results

### Model Comparison (Test Set — 2,000 samples)

| Model | Accuracy | Weighted F1 | **Macro F1** |
|---|---|---|---|
| Decision Tree (Baseline) | 97% | 0.96 | 0.28 |
| Random Forest | 98% | 0.98 | 0.52 |
| **XGBoost** | **98%** | **0.98** | **0.60** |

> ⚠️ **Macro F1** is the meaningful metric here — accuracy is misleadingly high due to class imbalance. A model predicting "No Failure" for everything would score 96.5%.

### XGBoost Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No Failure | 0.99 | 1.00 | **0.99** | 1930 |
| Heat Dissipation Failure | 0.95 | 0.86 | **0.90** | 22 |
| Overstrain Failure | 0.88 | 0.88 | **0.88** | 16 |
| Power Failure | 0.94 | 0.79 | **0.86** | 19 |
| Tool Wear Failure | 0.00 | 0.00 | 0.00 | 9 |
| Random Failures | 0.00 | 0.00 | 0.00 | 4 |

> **Tool Wear Failure** (45 samples) and **Random Failures** (18 samples) remain undetected — a genuine dataset limitation. Oversampling techniques like SMOTE or threshold tuning are natural next steps.

### Confusion Matrix & SHAP Summary

<div align="center">

| Confusion Matrix | SHAP Feature Importance |
|:---:|:---:|
| ![Confusion Matrix](confusion_matrix.png) | ![SHAP Summary](shap_summary.png) |

</div>

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) was applied to the XGBoost model using `shap.TreeExplainer` to understand **which sensor readings drive which failure predictions**.

- **Per-class feature importance** was computed across all 6 failure types
- The bar summary plot reveals the mean absolute SHAP contribution of each feature per class
- Key drivers expected: `Torque [Nm]`, `Tool wear [min]`, and `Rotational speed [rpm]` — directly tied to mechanical stress and wear patterns

This analysis makes the model actionable for maintenance engineers: rather than a black-box alert, they can see *why* a failure is predicted.

---

## 🗂️ Project Structure

```
predictive-maintenance-ml/
├── predictive_maintenance.ipynb   # Full ML pipeline notebook
├── predictive_maintenance.csv     # Dataset (10,000 samples)
├── confusion_matrix.png           # XGBoost confusion matrix
├── shap_summary.png               # SHAP feature importance plot
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

### Run the Notebook

```bash
jupyter notebook predictive_maintenance.ipynb
```

Or open directly in **IBM Watson Studio** (Cloud Lite) — the platform used for development per project requirements.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading & preprocessing |
| `scikit-learn` | Decision Tree, Random Forest, train/test split, metrics |
| `XGBoost` | Best-performing classifier |
| `SHAP` | Model explainability |
| `matplotlib` / `seaborn` | Visualizations |
| IBM Watson Studio | Cloud execution environment |

---

## 🔮 Future Improvements

- [ ] Apply **SMOTE** or **ADASYN** oversampling to improve minority class detection
- [ ] Tune XGBoost hyperparameters with `GridSearchCV` or `Optuna`
- [ ] Explore **threshold optimization** per class using precision-recall curves
- [ ] Add **SHAP waterfall / force plots** for individual prediction explanations
- [ ] Deploy model as a REST API for real-time sensor stream inference

---

<div align="center">

**AICTE-2026 · Problem Statement No. 39 · Mechanical Engineering ML Track**

</div>
