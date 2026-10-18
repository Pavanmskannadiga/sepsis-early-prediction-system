"""
Evaluation Metrics Module for the Sepsis Prediction System.

Provides comprehensive metrics for binary classification with emphasis
on metrics relevant to imbalanced clinical data: AUROC, AUPRC,
sensitivity/specificity, and the PhysioNet Utility Score.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground truth binary labels, shape (n,).
        y_prob: Predicted probabilities, shape (n,).
        threshold: Classification threshold.

    Returns:
        Dictionary of metric name → value.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Handle edge cases
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    metrics = {}

    # Discrimination metrics
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = 0.0

    try:
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auprc"] = 0.0

    # Classification metrics at threshold
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["sensitivity"] = recall_score(y_true, y_pred, zero_division=0)  # = recall
    metrics["recall"] = metrics["sensitivity"]
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive predictive value

    # Confusion matrix values
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tn"] = int(tn)

    # Matthews Correlation Coefficient (balanced metric for imbalanced data)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # Class distribution
    metrics["n_positive"] = int(n_pos)
    metrics["n_negative"] = int(n_neg)
    metrics["prevalence"] = float(n_pos / len(y_true)) if len(y_true) > 0 else 0.0
    metrics["threshold"] = threshold

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find the classification threshold that maximizes a given metric.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        metric: Metric to optimize ("f1", "youden", "sensitivity_90").

    Returns:
        Tuple of (optimal_threshold, best_metric_value).
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            # Youden's J statistic = sensitivity + specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sens + spec - 1
        elif metric == "sensitivity_90":
            # Threshold that gives >= 90% sensitivity with best specificity
            sens = recall_score(y_true, y_pred, zero_division=0)
            if sens >= 0.90:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                score = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                score = 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score


def get_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve data for plotting.

    Returns:
        Dict with 'fpr', 'tpr', 'thresholds', 'auc'.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc
    }


def get_pr_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute Precision-Recall curve data for plotting.
    PR-AUC is more informative than ROC-AUC for imbalanced data.

    Returns:
        Dict with 'precision', 'recall', 'thresholds', 'auc'.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    auc = average_precision_score(y_true, y_prob)
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "auc": auc
    }


def format_metrics_table(
    metrics: Dict[str, float],
    title: str = "Evaluation Metrics"
) -> str:
    """
    Format metrics as a readable table string.

    Args:
        metrics: Dictionary of metric → value.
        title: Table title.

    Returns:
        Formatted table string.
    """
    lines = [
        "=" * 50,
        f"  {title}",
        "=" * 50,
    ]

    key_order = [
        "auroc", "auprc", "f1", "precision", "sensitivity",
        "specificity", "accuracy", "mcc", "threshold"
    ]

    for key in key_order:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                lines.append(f"  {key:20s}: {val:.4f}")
            else:
                lines.append(f"  {key:20s}: {val}")

    lines.append("-" * 50)

    if "tp" in metrics:
        lines.append(f"  Confusion Matrix:")
        lines.append(f"    TP={metrics['tp']:5d}  FP={metrics['fp']:5d}")
        lines.append(f"    FN={metrics['fn']:5d}  TN={metrics['tn']:5d}")

    lines.append("=" * 50)
    return "\n".join(lines)
