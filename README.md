# Credit Default Timing Prediction (Multiclass Classification)

Predicting **not just whether** a customer will default on credit, but **when** — using machine learning on 450K+ customer financial records.

---

## Problem Statement

Traditional credit risk models are typically **binary** — they only predict *if* a customer will default or not. This project goes a step further by framing default prediction as a **multiclass classification problem**, predicting the **timing window** of a potential default. This gives lenders more actionable lead time to intervene (offer restructuring, adjust credit limits, early collections, etc.) before a default actually happens.

### Target Classes
| Class | Meaning |
|-------|---------|
| 0 | No Default |
| 1 | Default in 1–6 months |
| 2 | Default in 7–12 months |
| 3 | Default in 13–18 months |

---

## Dataset

- **Source**: [Default Time Prediction Multiclass — AmEx (Kaggle)](https://www.kaggle.com/datasets/arpan129/default-time-prediction-multiclass-amex)
- **Size**: 450,000+ customer records
- **Features**: 188 financial behavior features per customer
- **Files used**: `train_allx.csv` (features), `train_y.csv` (labels — `Default_Flag`), `val_allx.csv` (validation features)

> Note: The dataset is not included in this repo due to its size. Download it from the Kaggle link above and place it under `data/` (or update the `base_path` variable in the notebook) before running.

---

## Approach / Pipeline

1. **EDA** — class distribution analysis, feature distributions, correlation checks
2. **Preprocessing**
   - Label encoding for categorical columns
   - Missing value imputation using median
3. **Train/Validation Split** — stratified 80/20 split to preserve class balance
4. **Modeling** — three models trained and compared:
   - Logistic Regression (baseline)
   - XGBoost
   - LightGBM
5. **Evaluation** — Weighted One-vs-Rest (OvR) ROC-AUC, classification report, confusion matrix, per-class AUC
6. **Explainability** — SHAP values used to interpret the best model's predictions, both globally (feature importance) and for individual customer risk profiles
7. **Business Insights** — top risk-driving features and a sample early-warning customer risk profile

---

## Results

The best-performing model was selected based on **weighted OvR AUC** across all four classes. Detailed metrics (per-class AUC, precision, recall, F1) are available in the notebook output.

Key feature-importance and business-insight findings are summarized at the end of the notebook, including the top financial behavior features driving default-timing predictions.

---

## Tech Stack

- **Language**: Python
- **Data handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Modeling**: scikit-learn (Logistic Regression), XGBoost, LightGBM
- **Explainability**: SHAP
- **Environment**: Jupyter Notebook (originally run on Kaggle)

---

## Project Structure

```
credit-default-timing-prediction/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── credit-default-timing-prediction.ipynb
```

---

## How to Run

1. Clone the repo
   ```bash
   git clone https://github.com/<your-username>/credit-default-timing-prediction.git
   cd credit-default-timing-prediction
   ```

2. Create a virtual environment and install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/arpan129/default-time-prediction-multiclass-amex) and update the `base_path` variable in the notebook to point to your local data folder.

4. Open and run the notebook
   ```bash
   jupyter notebook notebooks/credit-default-timing-prediction.ipynb
   ```

---

## Future Improvements

- Hyperparameter tuning (Optuna / GridSearchCV) for XGBoost & LightGBM
- Handling class imbalance with SMOTE or class-weighting
- Feature selection based on SHAP importance to reduce dimensionality
- Deploying the best model as an API for real-time risk scoring

---

## Author

Created as a personal/portfolio data science project.
Feel free to fork, star ⭐, or open an issue with suggestions.
