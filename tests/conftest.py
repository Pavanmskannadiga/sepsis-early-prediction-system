"""
conftest.py — Shared pytest fixtures for the sepsis prediction test suite.

All fixtures are available to every test file in this directory without
needing any explicit import — pytest injects them automatically.
"""

import sys
sys.path.insert(0, 'd:/sp system')

import pytest
import torch
import numpy as np
from src.models.bilstm_attention import BiLSTMAttention


# ---------------------------------------------------------------------------
# Tensor / model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_batch():
    """
    A tiny 4-sample batch that matches the real model's input shape.

    Shape:
        X : (batch=4, seq_len=48, input_dim=68)
        y : (4,) binary labels, float32
    """
    X = torch.randn(4, 48, 68)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    return X, y


@pytest.fixture
def tiny_model():
    """
    A small BiLSTMAttention instance suitable for fast unit tests.

    Uses input_dim=68 (matching sample_batch) but a much smaller
    hidden_dim/fc_hidden_dim so tests complete quickly on CPU.
    """
    model = BiLSTMAttention(
        input_dim=68,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        fc_hidden_dim=16,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# NumPy label / probability fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_y_true():
    """Ground-truth binary labels (10 samples, ~40 % positive)."""
    return np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 0])


@pytest.fixture
def sample_y_prob():
    """Predicted probabilities that broadly agree with sample_y_true."""
    return np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.3, 0.7, 0.85, 0.25, 0.1])
