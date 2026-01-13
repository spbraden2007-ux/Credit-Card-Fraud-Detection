"""
Credit Card Fraud Detection - Ensemble Pipeline
================================================
EllipticEnvelope (unsupervised) + LightGBM (supervised) with Optuna optimization

Performance: 0.93 F1-macro (Dacon Leaderboard)
Dataset: 284,807 transactions, 0.17% fraud rate (1:588 imbalance)

Author: Seohyun Park
"""

import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import EllipticEnvelope
from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler
import torch
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Data Loading
# =============================================================================
train = pd.read_csv('data/train.csv').drop(['ID'], axis=1)
val = pd.read_csv('data/val.csv')
test = pd.read_csv('data/test.csv').drop(['ID'], axis=1)

fraud_ratio = val['Class'].sum() / len(val)
print(f"Fraud ratio: {fraud_ratio:.4f} ({fraud_ratio*100:.2f}%)")


# =============================================================================
# Step 1: Unsupervised Anomaly Detection
# =============================================================================
envelope = EllipticEnvelope(
    support_fraction=0.994,
    contamination=fraud_ratio,
    random_state=42
)
envelope.fit(train)


# =============================================================================
# Step 2: Pseudo-Label Generation
# =============================================================================
def generate_pseudo_labels(model, X, k):
    """
    Select top-k most anomalous samples as fraud candidates.
    Lower anomaly score = more suspicious.
    """
    scores = torch.tensor(model.score_samples(X), dtype=torch.float)
    top_k_idx = torch.topk(scores, k=k, largest=False).indices
    
    labels = torch.zeros(len(X), dtype=torch.long)
    labels[top_k_idx] = 1
    return labels.numpy(), scores.numpy()


# Generate pseudo-labels
train_labels, _ = generate_pseudo_labels(envelope, train, k=118)
test_pred_envelope, _ = generate_pseudo_labels(envelope, test, k=314)


# =============================================================================
# Step 3: Optuna Hyperparameter Optimization
# =============================================================================
def objective(trial):
    """5-fold CV optimization for macro F1-score."""
    params = {
        "boosting_type": trial.suggest_categorical('boosting_type', ['dart', 'gbdt']),
        "learning_rate": trial.suggest_float('learning_rate', 0.2, 0.99),
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=10),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 30),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "max_bin": trial.suggest_int("max_bin", 50, 100),
        "verbosity": -1,
        "random_state": 42
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(train, train_labels):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        scores.append(f1_score(y_val, pred, average='macro'))
    
    return np.mean(scores)


# Run optimization
print("Starting Optuna optimization (500 trials)...")
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42)
)
study.optimize(objective, n_trials=500, show_progress_bar=True)

print(f"\nBest F1-macro: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")


# =============================================================================
# Step 4: Train Final Model
# =============================================================================
# Pre-optimized parameters (from 500-trial search)
best_params = {
    'boosting_type': 'dart',
    'learning_rate': 0.3066,
    'n_estimators': 270,
    'max_depth': 7,
    'num_leaves': 66,
    'reg_alpha': 0.0531,
    'reg_lambda': 0.8492,
    'subsample': 0.5663,
    'subsample_freq': 1,
    'colsample_bytree': 0.9079,
    'min_child_samples': 31,
    'max_bin': 52,
    'verbosity': -1,
    'random_state': 2893
}

lgb_model = LGBMClassifier(**best_params)
lgb_model.fit(train, train_labels)
test_pred_lgb = lgb_model.predict(test)


# =============================================================================
# Step 5: Ensemble Voting (OR Logic)
# =============================================================================
# OR: Flag as fraud if EITHER model predicts fraud
# Rationale: False negatives (missed fraud) costlier than false positives
final_pred = test_pred_envelope | test_pred_lgb

# Save results
submission = pd.read_csv('data/sample_submission.csv')
submission['Class'] = final_pred
submission.to_csv('output/result.csv', index=False)

print(f"\n✓ Saved to output/result.csv")
print(f"✓ Fraud detected: {final_pred.sum()} / {len(final_pred)} ({final_pred.mean()*100:.2f}%)")
