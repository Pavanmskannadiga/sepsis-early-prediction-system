"""
test_model.py — Unit tests for BiLSTM + Multi-Head Attention architecture.

Covers:
    - Forward-pass output shape
    - Probability range (sigmoid output in [0, 1])
    - Attention-weight storage and retrieval
    - SimpleLSTM baseline forward pass
    - Parameter counting
    - Model summary string
    - build_model() factory function
    - Gradient flow through the LSTM encoder
"""

import sys
sys.path.insert(0, 'd:/sp system')

import pytest
import torch
import torch.nn as nn
from src.models.bilstm_attention import BiLSTMAttention, SimpleLSTM, build_model


# ---------------------------------------------------------------------------
# Forward-pass shape
# ---------------------------------------------------------------------------

def test_bilstm_forward_pass_shape(tiny_model, sample_batch):
    """Output logits must have shape (batch, 1)."""
    X, y = sample_batch
    with torch.no_grad():
        output = tiny_model(X)
    assert output.shape == (4, 1), (
        f"Expected output shape (4, 1), got {output.shape}"
    )


# ---------------------------------------------------------------------------
# Probability range
# ---------------------------------------------------------------------------

def test_bilstm_predict_proba_in_range(tiny_model, sample_batch):
    """After sigmoid, all probabilities must lie in [0, 1]."""
    X, _ = sample_batch
    probs = tiny_model.predict_proba(X)           # calls sigmoid internally
    assert probs.shape == (4, 1)
    assert torch.all(probs >= 0.0), "Probability below 0 found"
    assert torch.all(probs <= 1.0), "Probability above 1 found"


# ---------------------------------------------------------------------------
# Attention weights
# ---------------------------------------------------------------------------

def test_attention_weights_stored(tiny_model, sample_batch):
    """
    Calling forward with return_attention=True must store attention weights
    that are later retrievable via get_attention_weights().
    """
    X, _ = sample_batch
    with torch.no_grad():
        tiny_model(X, return_attention=True)
    attn = tiny_model.get_attention_weights()
    assert attn is not None, "Attention weights should not be None after forward(return_attention=True)"
    # Shape: (batch, num_heads, seq_len, seq_len)
    assert attn.dim() == 4, f"Expected 4-D attention tensor, got {attn.dim()}-D"


# ---------------------------------------------------------------------------
# SimpleLSTM baseline
# ---------------------------------------------------------------------------

def test_simple_lstm_forward():
    """SimpleLSTM must accept (batch, seq, features) and return (batch, 1)."""
    model = SimpleLSTM(input_dim=68, hidden_dim=32)
    x = torch.randn(2, 48, 68)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def test_model_parameter_count(tiny_model):
    """count_parameters() must return a dict with positive 'total'."""
    counts = tiny_model.count_parameters()
    assert isinstance(counts, dict), "count_parameters() must return a dict"
    assert "total" in counts, "Missing 'total' key"
    assert counts["total"] > 0, "'total' parameter count must be > 0"


# ---------------------------------------------------------------------------
# Summary string
# ---------------------------------------------------------------------------

def test_model_summary_returns_string(tiny_model):
    """summary() must return a non-empty string."""
    s = tiny_model.summary()
    assert isinstance(s, str), "summary() must return a str"
    assert len(s) > 0, "summary() must not return an empty string"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def test_build_model_factory():
    """build_model('bilstm_attention') must return a BiLSTMAttention instance."""
    model = build_model("bilstm_attention")
    assert isinstance(model, BiLSTMAttention), (
        f"Expected BiLSTMAttention, got {type(model)}"
    )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow(tiny_model, sample_batch):
    """
    After a forward + backward pass, the LSTM input-hidden weight matrix
    must have a non-None gradient, confirming end-to-end gradient flow.
    """
    tiny_model.train()          # enable gradient computation
    X, y = sample_batch
    output = tiny_model(X)      # (4, 1)
    # Use a simple BCE-style loss
    loss = nn.functional.binary_cross_entropy_with_logits(output.squeeze(1), y)
    loss.backward()
    grad = tiny_model.lstm.weight_ih_l0.grad
    assert grad is not None, "LSTM weight_ih_l0.grad should not be None after backward()"
