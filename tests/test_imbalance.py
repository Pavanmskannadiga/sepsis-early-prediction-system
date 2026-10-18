"""
test_imbalance.py — Unit tests for class imbalance handling utilities.

Covers:
    - FocalLoss outputs a scalar (0-dim tensor)
    - FocalLoss value is strictly positive
    - get_loss_function('focal') returns an nn.Module
    - get_loss_function('bce') returns an nn.Module
    - get_loss_function('weighted_bce', y_train=...) returns an nn.Module
    - get_class_weights() assigns higher weight to the minority class
    - apply_smote() preserves the number of features (skipped if imblearn absent)
"""

import sys
sys.path.insert(0, 'd:/sp system')

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.data.imbalance import FocalLoss, get_loss_function, get_class_weights


# ---------------------------------------------------------------------------
# FocalLoss — scalar output
# ---------------------------------------------------------------------------

def test_focal_loss_output_is_scalar():
    """FocalLoss should reduce to a 0-dimensional (scalar) tensor by default."""
    criterion = FocalLoss()
    logits  = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8, 1)).float()
    loss = criterion(logits, targets)
    assert loss.dim() == 0, (
        f"Expected scalar (0-D tensor), got {loss.dim()}-D tensor"
    )


# ---------------------------------------------------------------------------
# FocalLoss — positivity
# ---------------------------------------------------------------------------

def test_focal_loss_is_positive():
    """FocalLoss must always be strictly positive."""
    criterion = FocalLoss()
    logits  = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8, 1)).float()
    loss = criterion(logits, targets)
    assert loss.item() > 0.0, (
        f"Focal loss should be > 0, got {loss.item()}"
    )


# ---------------------------------------------------------------------------
# get_loss_function — focal
# ---------------------------------------------------------------------------

def test_get_loss_function_focal():
    """get_loss_function('focal') must return an nn.Module instance."""
    loss_fn = get_loss_function("focal")
    assert isinstance(loss_fn, nn.Module), (
        f"Expected nn.Module, got {type(loss_fn)}"
    )


# ---------------------------------------------------------------------------
# get_loss_function — bce
# ---------------------------------------------------------------------------

def test_get_loss_function_bce():
    """get_loss_function('bce') must return an nn.Module instance."""
    loss_fn = get_loss_function("bce")
    assert isinstance(loss_fn, nn.Module), (
        f"Expected nn.Module, got {type(loss_fn)}"
    )


# ---------------------------------------------------------------------------
# get_loss_function — weighted_bce
# ---------------------------------------------------------------------------

def test_get_loss_function_weighted_bce():
    """
    get_loss_function('weighted_bce', y_train=...) must return an nn.Module.
    Provides a simple imbalanced label array: 4 negatives and 2 positives.
    """
    y_train = np.array([0, 0, 0, 0, 1, 1])
    loss_fn = get_loss_function("weighted_bce", y_train=y_train)
    assert isinstance(loss_fn, nn.Module), (
        f"Expected nn.Module, got {type(loss_fn)}"
    )


# ---------------------------------------------------------------------------
# get_class_weights — minority class gets higher weight
# ---------------------------------------------------------------------------

def test_class_weights_minority_higher():
    """
    With 4 negatives and 1 positive, the positive (minority) class weight
    must exceed the implicit negative weight (1.0).

    get_class_weights() returns pos_weight = n_neg / n_pos, so for
    y = [0, 0, 0, 0, 1] → pos_weight = 4/1 = 4.0 > 1.0.
    """
    y = np.array([0, 0, 0, 0, 1])
    pos_weight = get_class_weights(y)               # tensor([pos_weight])
    # The returned tensor is the weight for the positive class;
    # a value > 1.0 confirms minority class is weighted higher.
    assert pos_weight.item() > 1.0, (
        f"Expected pos_weight > 1.0 (minority class), got {pos_weight.item()}"
    )


# ---------------------------------------------------------------------------
# apply_smote — feature dimension preserved (skipped if imblearn absent)
# ---------------------------------------------------------------------------

def test_smote_shape(sample_batch):
    """
    apply_smote() on 2-D data must preserve the number of features.
    Skipped if imbalanced-learn is not installed.
    """
    pytest.importorskip(
        "imblearn",
        reason="imbalanced-learn not installed; skipping SMOTE shape test",
    )
    from src.data.imbalance import apply_smote

    # Use 2-D data to avoid reshape issues in a minimal test
    X, y = sample_batch
    # Collapse to 2D: (batch=4, seq_len*features)
    X_2d = X.numpy().reshape(4, -1)
    y_np = y.numpy().astype(int)

    # Need at least 2 positives for SMOTE with k_neighbors=1
    y_np = np.array([0, 1, 0, 1])   # 2 positives
    X_2d = np.random.randn(4, 68).astype(np.float32)

    n_features_before = X_2d.shape[1]
    X_res, y_res = apply_smote(X_2d, y_np, k_neighbors=1)
    n_features_after = X_res.shape[1]

    assert n_features_before == n_features_after, (
        f"Feature count changed after SMOTE: {n_features_before} → {n_features_after}"
    )
