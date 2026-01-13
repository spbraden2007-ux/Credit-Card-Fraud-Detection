# Credit Card Fraud Detection — 0.93 Public F1-macro (Dacon)

Hybrid anomaly→pseudo-label→LightGBM(Optuna) pipeline for extreme class imbalance (0.17% fraud).  
EllipticEnvelope provides anomaly scores → pseudo-labels create training signal → OR-ensemble prioritizes recall.

> Research Internship (Jeonju University AI Lab, Jan–Feb 2024)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-green.svg)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-3.0+-blue.svg)](https://optuna.org/)

---

## Performance

| Metric | Value |
|--------|-------|
| Dacon Public Leaderboard | **0.93 F1-macro** |
| Cross-Validation (5-fold) | 0.87 ± 0.006 (F1-macro) |
| Class Imbalance Ratio | 1:588 (0.17% fraud) |
| Dataset Size | 284,807 transactions |

---

## Architecture

```
Raw Data (284K transactions)
         │
         ▼
┌─────────────────────────────┐
│  EllipticEnvelope           │  Unsupervised anomaly scoring
│  contamination=0.0017       │  (Gaussian outlier detection)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Pseudo-Label Generation    │  Top-k selection where k =
│  k = expected fraud count   │  expected fraud count per split
│  based on val fraud rate    │  (validation fraud rate: 0.17%)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Optuna + LightGBM          │  500 trials × 5-fold CV
│  Objective: Macro F1        │  TPE Bayesian optimization
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Ensemble (OR logic)        │  Prioritize recall over precision
│  final = envelope | lgb     │  (threshold adjustable to capacity)
└─────────────────────────────┘
```

---

## Quick Start

```bash
git clone https://github.com/spbraden2007-ux/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
python model.py
```

---

## Key Technical Decisions

### Pseudo-Labeling Strategy
Ground truth unavailable for test set. EllipticEnvelope identifies statistical outliers, and top-k selection creates training signal for supervised model without SMOTE artifacts.

### k Calibration
k values were set to the expected fraud count per split based on validation fraud rate (0.17%). This maintains consistent class distribution across training splits.

### Why LightGBM + DART?
- Handles tabular data better than neural networks at this scale
- DART (Dropout Additive Regression Trees) prevents overfitting
- Native handling of imbalanced classes via leaf-wise growth

### Why OR Ensemble?
Fraud detection prioritizes recall—false negatives (missed fraud) are costlier than false positives (manual review). Chosen to maximize detection coverage; thresholding can be adjusted to match investigation capacity.

---

## Optimized Hyperparameters

```python
{
    'boosting_type': 'dart',
    'learning_rate': 0.307,
    'n_estimators': 270,
    'max_depth': 7,
    'num_leaves': 66,
    'reg_alpha': 0.053,
    'reg_lambda': 0.849,
    'subsample': 0.566,
    'colsample_bytree': 0.908,
    'min_child_samples': 31
}
```

---

## Results

### Cross-Validation Stability

| Fold | F1-macro |
|------|----------|
| 1 | 0.8691 |
| 2 | 0.8798 |
| 3 | 0.8645 |
| 4 | 0.8723 |
| 5 | 0.8757 |
| **Mean** | **0.8723 ± 0.0058** |

### Model Comparison

| Model | CV F1 | Public F1 |
|-------|-------|-----------|
| EllipticEnvelope only | 0.78 | 0.85 |
| LightGBM only | 0.78 | 0.88 |
| **Ensemble (OR)** | **0.87** | **0.93** |

---

## Limitations & Future Work

| Limitation | Potential Improvement |
|------------|----------------------|
| Fixed k-value | Auto-calibrate via PR curve |
| No temporal features | Add transaction velocity |
| No explainability | SHAP values for flagged cases |

---

## Project Structure

```
Credit-Card-Fraud-Detection/
├── model.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

Dataset (284K rows) excluded. Available at [Dacon Competition](https://dacon.io/competitions/official/235930/data).

---

## References

- Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Akiba et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD.

---

**Seohyun Park** · University of Waterloo, Computer Science  
*Supervised by Professor Sunwoo Ko*
