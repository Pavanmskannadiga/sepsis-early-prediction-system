"""
Baseline Models for Sepsis Prediction.

Provides classical ML baselines (Logistic Regression, Random Forest,
XGBoost, LightGBM) for comparison with the BiLSTM+Attention model.
These models operate on flattened or aggregated feature representations.

For tree-ensemble baselines, we compute aggregate statistics over the
time window (mean, std, min, max, last, trend) to create a fixed-length
feature vector from variable-length sequences.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

# Conditional imports for optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def extract_aggregate_features(X: np.ndarray) -> np.ndarray:
    """
    Extract statistical aggregates from time-series sequences to create
    fixed-length feature vectors for classical ML models.

    For each of the F features across T timesteps, computes:
      - mean, std, min, max, last value, linear trend slope
    Resulting in 6*F features per sample.

    Args:
        X: Sequence data, shape (n_samples, seq_len, n_features).

    Returns:
        Aggregated feature matrix, shape (n_samples, n_features * 6).
    """
    n_samples, seq_len, n_features = X.shape

    # Statistical aggregates
    feat_mean = np.nanmean(X, axis=1)
    feat_std = np.nanstd(X, axis=1)
    feat_min = np.nanmin(X, axis=1)
    feat_max = np.nanmax(X, axis=1)
    feat_last = X[:, -1, :]

    # Linear trend (slope via least squares)
    t = np.arange(seq_len).reshape(1, -1, 1)  # (1, T, 1)
    t_mean = t.mean(axis=1, keepdims=True)
    x_mean = np.nanmean(X, axis=1, keepdims=True)
    numerator = np.nanmean((t - t_mean) * (X - x_mean), axis=1)
    denominator = np.nanmean((t - t_mean) ** 2, axis=1) + 1e-8
    feat_trend = numerator / denominator

    # Concatenate all aggregates
    aggregated = np.concatenate([
        feat_mean, feat_std, feat_min, feat_max, feat_last, feat_trend
    ], axis=1)

    # Replace any remaining NaN/Inf with 0
    aggregated = np.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)

    return aggregated.astype(np.float32)


def get_baseline_models(random_state: int = 42) -> Dict[str, object]:
    """
    Create a dictionary of baseline classification models.

    Returns:
        Dictionary mapping model name to sklearn-compatible classifier.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
            C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            subsample=0.8
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            scale_pos_weight=12,  # ~1/0.08 for sepsis prevalence
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            is_unbalance=True,
            n_jobs=-1,
            verbose=-1
        )

    return models


def train_and_evaluate_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    aggregate: bool = True,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate all baseline models.

    Args:
        X_train: Training features (3D sequences or 2D flat).
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        aggregate: If True and X is 3D, compute aggregate features.
        verbose: Print results.

    Returns:
        Dictionary mapping model name to metrics dict.
    """
    # Convert 3D sequences to 2D aggregate features if needed
    if aggregate and X_train.ndim == 3:
        if verbose:
            print("Extracting aggregate features from sequences...")
        X_train_flat = extract_aggregate_features(X_train)
        X_test_flat = extract_aggregate_features(X_test)
    else:
        X_train_flat = X_train
        X_test_flat = X_test

    if verbose:
        print(f"Training features shape: {X_train_flat.shape}")
        print(f"Test features shape: {X_test_flat.shape}")
        print()

    models = get_baseline_models()
    results = {}

    for name, model in models.items():
        if verbose:
            print(f"Training {name}...", end=" ")

        try:
            model.fit(X_train_flat, y_train)

            # Predictions
            y_pred = model.predict(X_test_flat)
            y_prob = model.predict_proba(X_test_flat)[:, 1]

            # Metrics
            metrics = {
                "auroc": roc_auc_score(y_test, y_prob),
                "f1": f1_score(y_test, y_pred),
                "sensitivity": recall_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
            }

            results[name] = metrics

            if verbose:
                print(f"AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}, "
                      f"Sens={metrics['sensitivity']:.4f}")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results[name] = {"error": str(e)}

    return results
