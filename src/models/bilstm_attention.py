"""
BiLSTM + Multi-Head Attention Model for Sepsis Prediction.

This is the core model architecture implementing Research Objective O1:
predict sepsis onset at least 6 hours in advance with AUC >= 0.90.

Architecture:
    Input (batch, seq_len=48, features=68)
        ↓
    BiLSTM Layer 1 (hidden=128, bidirectional → 256)
        ↓ + LayerNorm + Dropout(0.3)
    BiLSTM Layer 2 (hidden=128, bidirectional → 256)
        ↓
    Multi-Head Self-Attention (4 heads, dim=256)
        ↓
    Global Max Pooling ⊕ Attention-Weighted Context
        ↓
    FC Layer (512 → 128) + GELU + LayerNorm + Dropout(0.3)
        ↓
    FC Layer (128 → 1)
        ↓
    Output: logit (apply sigmoid for probability)

Design rationale:
    - BiLSTM captures both forward and backward temporal dependencies
      in the observation window (past trends AND future context).
    - Multi-head attention learns WHICH timesteps are most predictive
      of sepsis, aligning with clinical intuition that acute deterioration
      patterns (e.g., sudden BP drop + lactate spike) are more important
      than stable observations.
    - Combining attention-weighted context with max-pooled features
      captures both the "most attended" and "most extreme" signals.
    - LayerNorm (instead of BatchNorm) for more stable training with
      variable-length sequences.

Authors: Pavana M.S., Megharaja S., Srujan S. Sagar, Abubakkar Siddiq
Guide: Prof. Shilpa K.C.
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for temporal sequences.

    Computes scaled dot-product attention across time steps to identify
    which hours of patient observation are most predictive. Each attention
    head can learn different temporal patterns (e.g., one head may attend
    to recent vital sign changes, another to lab trend patterns).

    Args:
        embed_dim: Input embedding dimension (= BiLSTM output dim).
        num_heads: Number of parallel attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Query, Key, Value projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, embed_dim).
            return_attention: If True, return attention weights for
                              interpretability.

        Returns:
            Tuple of (output, attention_weights).
            output shape: (batch, seq_len, embed_dim)
            attention_weights shape: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (B, T, D) → (B, H, T, D/H)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, D/H)

        # Reshape back: (B, H, T, D/H) → (B, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.W_o(attn_output)

        if return_attention:
            return output, attn_weights
        return output, None


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Multi-Head Self-Attention for early sepsis
    prediction.

    This model processes hourly clinical time series data (vital signs,
    lab values, and missingness masks) through a 2-layer BiLSTM to capture
    temporal dependencies, then applies multi-head self-attention to weight
    the most informative timesteps, and finally classifies the risk of
    sepsis onset within 6 hours.

    Args:
        input_dim: Number of input features per timestep.
        hidden_dim: LSTM hidden dimension (per direction).
        num_layers: Number of BiLSTM layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        fc_hidden_dim: Hidden dimension of the classification head.
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        num_heads: int = None,
        dropout: float = None,
        fc_hidden_dim: int = None
    ):
        super().__init__()

        # Use config defaults if not specified
        mc = config.ModelConfig
        self.input_dim = input_dim or mc.input_dim
        self.hidden_dim = hidden_dim or mc.hidden_dim
        self.num_layers = num_layers or mc.num_layers
        self.num_heads = num_heads or mc.num_attention_heads
        self.dropout_rate = dropout or mc.dropout
        self.fc_hidden_dim = fc_hidden_dim or mc.fc_hidden_dim

        # BiLSTM output dimension (bidirectional → 2x hidden)
        self.lstm_output_dim = self.hidden_dim * 2

        # ---- BiLSTM Encoder ----
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0
        )

        # Layer normalization after LSTM (more stable than BatchNorm for sequences)
        self.layer_norm = nn.LayerNorm(self.lstm_output_dim)

        # ---- Multi-Head Self-Attention ----
        self.attention = MultiHeadAttention(
            embed_dim=self.lstm_output_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate
        )

        self.attention_norm = nn.LayerNorm(self.lstm_output_dim)

        # ---- Classification Head ----
        # Combine attention context + max-pooled features → 2x lstm_output_dim
        classifier_input_dim = self.lstm_output_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.fc_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.fc_hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_hidden_dim, 1)
        )

        # Store attention weights for interpretability
        self._attention_weights = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization for linear layers."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # LSTM-specific: initialize forget gate bias to 1.0
        # (helps with gradient flow through long sequences)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                # Forget gate is the second quarter of the bias vector
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_dim).
            return_attention: If True, store attention weights for
                              later retrieval (for SHAP / interpretability).

        Returns:
            Logits tensor, shape (batch, 1). Apply sigmoid for probability.
        """
        # ---- BiLSTM encoding ----
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_dim*2)

        lstm_out = self.layer_norm(lstm_out)

        # ---- Multi-Head Self-Attention ----
        attn_out, attn_weights = self.attention(lstm_out, return_attention=True)
        # Residual connection + layer norm
        attn_out = self.attention_norm(attn_out + lstm_out)

        if return_attention:
            self._attention_weights = attn_weights

        # ---- Pooling: Attention-weighted context + Global Max Pool ----
        # Attention-weighted context vector (mean over attended timesteps)
        context = attn_out.mean(dim=1)  # (batch, lstm_output_dim)

        # Global max pooling over time (captures most extreme signal)
        max_pool, _ = lstm_out.max(dim=1)  # (batch, lstm_output_dim)

        # Concatenate both representations
        combined = torch.cat([context, max_pool], dim=1)  # (batch, lstm_output_dim*2)

        # ---- Classification ----
        logits = self.classifier(combined)  # (batch, 1)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities (applies sigmoid to logits).

        Args:
            x: Input tensor.

        Returns:
            Probability tensor, shape (batch, 1), values in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Retrieve stored attention weights from the last forward pass.
        Must call forward() with return_attention=True first.

        Returns:
            Attention weights, shape (batch, num_heads, seq_len, seq_len),
            or None if forward hasn't been called with return_attention=True.
        """
        return self._attention_weights

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters by component.

        Returns:
            Dictionary mapping component name to parameter count.
        """
        counts = {
            "lstm": sum(p.numel() for n, p in self.named_parameters() if "lstm" in n),
            "attention": sum(p.numel() for n, p in self.named_parameters() if "attention" in n and "norm" not in n),
            "classifier": sum(p.numel() for n, p in self.named_parameters() if "classifier" in n),
            "layer_norm": sum(p.numel() for n, p in self.named_parameters() if "norm" in n),
        }
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts

    def summary(self) -> str:
        """Human-readable model summary."""
        params = self.count_parameters()
        lines = [
            "=" * 60,
            "BiLSTM + Multi-Head Attention — Sepsis Prediction Model",
            "=" * 60,
            f"  Input dim:        {self.input_dim}",
            f"  Hidden dim:       {self.hidden_dim} (×2 bidirectional = {self.lstm_output_dim})",
            f"  LSTM layers:      {self.num_layers}",
            f"  Attention heads:  {self.num_heads}",
            f"  FC hidden dim:    {self.fc_hidden_dim}",
            f"  Dropout:          {self.dropout_rate}",
            "-" * 60,
            f"  LSTM params:      {params['lstm']:,}",
            f"  Attention params: {params['attention']:,}",
            f"  Classifier params:{params['classifier']:,}",
            f"  LayerNorm params: {params['layer_norm']:,}",
            "-" * 60,
            f"  Total params:     {params['total']:,}",
            f"  Trainable params: {params['trainable']:,}",
            "=" * 60,
        ]
        return "\n".join(lines)


class SimpleLSTM(nn.Module):
    """
    Standard unidirectional LSTM baseline (no attention).
    Used for comparison with the BiLSTM+Attention model.
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim or config.ModelConfig.input_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)


def build_model(model_type: str = "bilstm_attention", **kwargs) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        model_type: "bilstm_attention", "simple_lstm", "transformer", "tcn", or "ensemble".
        **kwargs: Model-specific keyword arguments.

    Returns:
        PyTorch model instance.
    """
    if model_type == "bilstm_attention":
        model = BiLSTMAttention(**kwargs)
    elif model_type == "simple_lstm":
        model = SimpleLSTM(**kwargs)
    elif model_type == "transformer":
        from src.models.transformer_model import TransformerSepsisModel
        model = TransformerSepsisModel(**kwargs)
    elif model_type == "tcn":
        from src.models.tcn_model import TCNSepsisModel
        model = TCNSepsisModel(**kwargs)
    elif model_type in ("bilstm_tcn", "tcn_bilstm"):
        from src.models.bilstm_tcn import BiLSTMTCN
        model = BiLSTMTCN(**kwargs)
    elif model_type == "ensemble":
        from src.models.ensemble import DeepSepsisEnsemble
        model = DeepSepsisEnsemble(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(model.summary() if hasattr(model, "summary") else model)
    return model
