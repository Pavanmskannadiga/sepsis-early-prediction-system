"""
test_metrics.py — Unit tests for the evaluation metrics module.

Covers:
    - Perfect AUROC on a trivially separable case
    - F1 score on hand-calculated ground truth
    - Sensitivity & specificity present and in [0, 1]
    - Optimal-threshold finder (F1 criterion)
    - ROC curve dictionary keys
    - PR curve dictionary keys
    - format_metrics_table() returning a string with 'auroc'
    - MCC range [-1, 1]
"""

import sys
sys.path.insert(0, 'd:/sp system')

import pytest
import numpy as np
from src.training.metrics import (
    compute_all_metrics,
    find_optimal_threshold,
    get_roc_curve_data,
    get_pr_curve_data,
    format_metrics_table,
)


# ---------------------------------------------------------------------------
# AUROC — perfect prediction
# ---------------------------------------------------------------------------

def test_auroc_perfect_prediction():
    """Perfectly separated scores must yield AUROC == 1.0."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = compute_all_metrics(y_true, y_prob)
    assert metrics["auroc"] == pytest.approx(1.0, abs=1e-6), (
        f"Expected AUROC=1.0, got {metrics['auroc']}"
    )


# ---------------------------------------------------------------------------
# F1 score — known hand-calculated value
# ---------------------------------------------------------------------------

def test_f1_computation():
    """
    Hand-calculated case:
        y_true = [1, 1, 0, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1]   → all predictions correct at 0.5
        TP=2, FP=0, FN=0 → precision=1, recall=1 → F1=1.0
    """
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["f1"] == pytest.approx(1.0, abs=1e-6), (
        f"Expected F1=1.0, got {metrics['f1']}"
    )


# ---------------------------------------------------------------------------
# Sensitivity and specificity
# ---------------------------------------------------------------------------

def test_sensitivity_and_specificity(sample_y_true, sample_y_prob):
    """Both 'sensitivity' and 'specificity' must exist and lie in [0, 1]."""
    metrics = compute_all_metrics(sample_y_true, sample_y_prob)
    assert "sensitivity" in metrics, "'sensitivity' key missing from metrics"
    assert "specificity" in metrics, "'specificity' key missing from metrics"
    assert 0.0 <= metrics["sensitivity"] <= 1.0, (
        f"sensitivity={metrics['sensitivity']} out of [0,1]"
    )
    assert 0.0 <= metrics["specificity"] <= 1.0, (
        f"specificity={metrics['specificity']} out of [0,1]"
    )


# ---------------------------------------------------------------------------
# Optimal threshold (F1)
# ---------------------------------------------------------------------------

def test_optimal_threshold_f1(sample_y_true, sample_y_prob):
    """
    find_optimal_threshold() must return a (threshold, score) tuple where
    threshold is a float strictly between 0 and 1.
    """
    result = find_optimal_threshold(sample_y_true, sample_y_prob, metric="f1")
    assert isinstance(result, tuple) and len(result) == 2, (
        "Expected a 2-tuple (threshold, score)"
    )
    threshold, score = result
    assert 0.0 < threshold < 1.0, (
        f"Optimal threshold {threshold} should be in open interval (0, 1)"
    )
    assert 0.0 <= score <= 1.0, (
        f"Optimal F1 score {score} should be in [0, 1]"
    )


# ---------------------------------------------------------------------------
# ROC curve data keys
# ---------------------------------------------------------------------------

def test_roc_curve_data_has_required_keys(sample_y_true, sample_y_prob):
    """get_roc_curve_data() must return a dict containing 'fpr', 'tpr', 'auc'."""
    result = get_roc_curve_data(sample_y_true, sample_y_prob)
    for key in ("fpr", "tpr", "auc"):
        assert key in result, f"Missing key '{key}' in ROC curve data"


# ---------------------------------------------------------------------------
# PR curve data keys
# ---------------------------------------------------------------------------

def test_pr_curve_data_has_required_keys(sample_y_true, sample_y_prob):
    """get_pr_curve_data() must return a dict containing 'precision', 'recall', 'auc'."""
    result = get_pr_curve_data(sample_y_true, sample_y_prob)
    for key in ("precision", "recall", "auc"):
        assert key in result, f"Missing key '{key}' in PR curve data"


# ---------------------------------------------------------------------------
# format_metrics_table
# ---------------------------------------------------------------------------

def test_format_metrics_table_returns_string(sample_y_true, sample_y_prob):
    """
    format_metrics_table() must return a non-empty string that contains
    the word 'auroc' (case-insensitive).
    """
    metrics = compute_all_metrics(sample_y_true, sample_y_prob)
    table = format_metrics_table(metrics)
    assert isinstance(table, str), "format_metrics_table() must return a str"
    assert len(table) > 0, "format_metrics_table() must not return an empty string"
    assert "auroc" in table.lower(), (
        "'auroc' not found in formatted metrics table"
    )


# ---------------------------------------------------------------------------
# MCC range
# ---------------------------------------------------------------------------

def test_mcc_range(sample_y_true, sample_y_prob):
    """Matthews Correlation Coefficient must be in [-1, 1]."""
    metrics = compute_all_metrics(sample_y_true, sample_y_prob)
    assert "mcc" in metrics, "'mcc' key missing from metrics"
    assert -1.0 <= metrics["mcc"] <= 1.0, (
        f"MCC={metrics['mcc']} is outside [-1, 1]"
    )
