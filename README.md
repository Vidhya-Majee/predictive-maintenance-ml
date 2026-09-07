# Predictive Maintenance of Industrial Machinery

ML classification model to predict machine failure types (Heat Dissipation, Power, Overstrain, Tool Wear, Random Failures) using sensor data.

## Problem Statement
AICTE-2026 Problem Statement No.39 — Mechanical Engineering ML Track

## Dataset
[Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (Kaggle)

## Approach
- Baseline: Decision Tree (max_depth=3)
- Random Forest (class_weight='balanced')
- XGBoost (best performing model)
- Stratified train/test split to handle severe class imbalance
- SHAP explainability analysis

## Results
XGBoost outperformed baseline and Random Forest, achieving strong F1 scores on Heat Dissipation (0.90), Overstrain (0.88), and Power Failure (0.86) classes. Random Failures and Tool Wear Failure remained undetectable due to insufficient training samples (14 and 36 respectively) — a genuine dataset limitation.

## Technology
IBM Cloud Lite (Watson Studio) — notebook executed on IBM's platform per project requirements.

## Files
- `predictive_maintenance.ipynb` — full notebook
- `predictive_maintenance.csv` — dataset
- `confusion_matrix.png`, `shap_summary.png` — visualizations