"""
Transformer Encoder Model for Sepsis Prediction.

Alternative deep learning architecture implementing a Transformer Encoder
to predict sepsis onset at least 6 hours in advance with AUC >= 0.90.

Architecture:
    Input (batch, seq_len=48, features=68)
        ↓
    Linear Projection (68 → d_model=256)
        ↓ + LayerNorm
    Sinusoidal Positional Encoding
        ↓
    Transformer Encoder (4 layers, 4 heads, d_ff=512, dropout=0.3)
        ↓
    Global Average Pooling ⊕ CLS Token Representation
        ↓
    FC Layer (d_model*2 → 128) + GELU + LayerNorm + Dropout(0.3)
        ↓
    FC Layer (128 → 1)
        ↓
    Output: logit (apply sigmoid for probability)

Design rationale:
    - The Transformer Encoder leverages global self-attention to capture
      long-range temporal dependencies without the sequential bottleneck
      of RNNs. Every timestep attends to every other timestep directly.
    - A learnable CLS token aggregates sequence-level information,
      complemented by global average pooling for robust representation.
    - Sinusoidal positional encoding injects temporal ordering without
      additional learnable parameters.
    - Pre-LayerNorm architecture improves gradient flow and training
      stability, especially for smaller datasets.

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


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.

    Injects position information into the input embeddings so the
    Transformer can distinguish between observations at different
    hours. Uses the standard sin/cos formulation from Vaswani et al.

    Args:
        d_model: Model embedding dimension.
        max_len: Maximum sequence length supported.
        dropout: Dropout applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encoding matrix once
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter, but moves with model to device)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Positionally encoded tensor, same shape as input.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSepsisModel(nn.Module):
    """
    Transformer Encoder for early sepsis prediction.

    Processes hourly clinical time series data through a stack of
    Transformer Encoder layers with multi-head self-attention, then
    aggregates the temporal representation for binary classification.

    Args:
        input_dim: Number of input features per timestep.
        d_model: Transformer model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer Encoder layers.
        d_ff: Feed-forward network hidden dimension.
        dropout: Dropout rate.
        fc_hidden_dim: Hidden dimension of the classification head.
        max_seq_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int = None,
        d_model: int = None,
        num_heads: int = None,
        num_layers: int = None,
        d_ff: int = None,
        dropout: float = None,
        fc_hidden_dim: int = None,
        max_seq_len: int = None
    ):
        super().__init__()

        # Use config defaults if not specified
        tc = config.TransformerConfig
        self.input_dim = input_dim or tc.input_dim
        self.d_model = d_model or tc.d_model
        self.num_heads = num_heads or tc.num_heads
        self.num_layers = num_layers or tc.num_layers
        self.d_ff = d_ff or tc.d_ff
        self.dropout_rate = dropout or tc.dropout
        self.fc_hidden_dim = fc_hidden_dim or tc.fc_hidden_dim
        self.max_seq_len = max_seq_len or tc.max_seq_len

        # ---- Input Projection ----
        # Project raw features to d_model dimension
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        # ---- Positional Encoding ----
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=self.d_model,
            max_len=self.max_seq_len,
            dropout=self.dropout_rate
        )

        # ---- Learnable CLS Token ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )

        # ---- Classification Head ----
        # Combine CLS token + global average pooling → 2 * d_model
        classifier_input_dim = self.d_model * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.fc_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.fc_hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_hidden_dim, 1)
        )

        # Store attention weights for interpretability
        self._attention_weights = None
        self._attention_hooks = []

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization for linear layers."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Initialize CLS token
        nn.init.normal_(self.cls_token, std=0.02)

    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights from encoder layers."""
        # Remove any existing hooks
        for hook in self._attention_hooks:
            hook.remove()
        self._attention_hooks = []

        self._attention_maps = []

        def make_hook():
            def hook_fn(module, input, output):
                # For TransformerEncoderLayer with batch_first=True,
                # we need to capture the attention weights from the
                # self-attention sublayer
                pass
            return hook_fn

        # We'll use a different approach: manually compute attention
        # in the forward pass when needed

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
        batch_size, seq_len, _ = x.shape

        # ---- Input Projection ----
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # ---- Prepend CLS Token ----
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # ---- Positional Encoding ----
        x = self.positional_encoding(x)  # (batch, seq_len+1, d_model)

        # ---- Transformer Encoder ----
        if return_attention:
            # Manually compute attention for interpretability
            attn_weights = self._forward_with_attention(x)
        else:
            x = self.transformer_encoder(x)  # (batch, seq_len+1, d_model)
            attn_weights = None

            # ---- Pooling ----
            cls_output = x[:, 0, :]          # CLS token: (batch, d_model)
            avg_pool = x[:, 1:, :].mean(dim=1)  # Avg pool over sequence: (batch, d_model)

            # Concatenate
            combined = torch.cat([cls_output, avg_pool], dim=1)  # (batch, d_model*2)

            # ---- Classification ----
            logits = self.classifier(combined)  # (batch, 1)
            return logits

        # When return_attention is True, x has been updated in _forward_with_attention
        # and self._last_encoded is set
        x_encoded = self._last_encoded

        cls_output = x_encoded[:, 0, :]
        avg_pool = x_encoded[:, 1:, :].mean(dim=1)
        combined = torch.cat([cls_output, avg_pool], dim=1)
        logits = self.classifier(combined)

        return logits

    def _forward_with_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder layers, capturing attention weights.

        Args:
            x: Input after projection and positional encoding,
               shape (batch, seq_len+1, d_model).

        Returns:
            Attention weights from the last layer,
            shape (batch, num_heads, seq_len+1, seq_len+1).
        """
        all_attn_weights = []

        for layer in self.transformer_encoder.layers:
            # Extract self-attention sublayer
            # TransformerEncoderLayer uses self_attn (nn.MultiheadAttention)
            # We manually call it to get attention weights

            # Pre-norm (norm_first=True)
            x_norm = layer.norm1(x)

            # Self-attention with attention weights
            attn_output, attn_weights = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False  # Get per-head weights
            )
            x = x + layer.dropout1(attn_output)

            # Feed-forward
            x_norm2 = layer.norm2(x)
            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x_norm2)))
            )
            x = x + layer.dropout2(ff_output)

            all_attn_weights.append(attn_weights)

        # Apply final norm
        if self.transformer_encoder.norm is not None:
            x = self.transformer_encoder.norm(x)

        # Store last layer attention weights
        self._attention_weights = all_attn_weights[-1]  # (batch, num_heads, seq+1, seq+1)
        self._last_encoded = x

        return all_attn_weights[-1]

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
            Attention weights, shape (batch, num_heads, seq_len+1, seq_len+1),
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
            "input_projection": sum(
                p.numel() for n, p in self.named_parameters()
                if "input_projection" in n
            ),
            "positional_encoding": sum(
                p.numel() for n, p in self.named_parameters()
                if "positional_encoding" in n
            ),
            "cls_token": self.cls_token.numel(),
            "transformer_encoder": sum(
                p.numel() for n, p in self.named_parameters()
                if "transformer_encoder" in n
            ),
            "classifier": sum(
                p.numel() for n, p in self.named_parameters()
                if "classifier" in n
            ),
        }
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return counts

    def summary(self) -> str:
        """Human-readable model summary."""
        params = self.count_parameters()
        lines = [
            "=" * 60,
            "Transformer Encoder — Sepsis Prediction Model",
            "=" * 60,
            f"  Input dim:            {self.input_dim}",
            f"  Model dim (d_model):  {self.d_model}",
            f"  Attention heads:      {self.num_heads}",
            f"  Encoder layers:       {self.num_layers}",
            f"  FF dim (d_ff):        {self.d_ff}",
            f"  FC hidden dim:        {self.fc_hidden_dim}",
            f"  Dropout:              {self.dropout_rate}",
            "-" * 60,
            f"  Input proj params:    {params['input_projection']:,}",
            f"  CLS token params:     {params['cls_token']:,}",
            f"  Encoder params:       {params['transformer_encoder']:,}",
            f"  Classifier params:    {params['classifier']:,}",
            "-" * 60,
            f"  Total params:         {params['total']:,}",
            f"  Trainable params:     {params['trainable']:,}",
            "=" * 60,
        ]
        return "\n".join(lines)
