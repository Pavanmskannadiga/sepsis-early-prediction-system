"""
Temporal Convolutional Network (TCN) Model for Sepsis Prediction.

Alternative deep learning architecture implementing a TCN with causal
dilated convolutions to predict sepsis onset at least 6 hours in advance
with AUC >= 0.90.

Architecture:
    Input (batch, seq_len=48, features=68)
        ↓
    TemporalBlock 1 (in=68, out=128, kernel=3, dilation=1)
      → CausalConv1d + WeightNorm + ReLU + Dropout
      → CausalConv1d + WeightNorm + ReLU + Dropout
      → Residual Connection (with 1×1 conv if dims differ)
        ↓
    TemporalBlock 2 (in=128, out=128, kernel=3, dilation=2)
        ↓
    TemporalBlock 3 (in=128, out=128, kernel=3, dilation=4)
        ↓
    TemporalBlock 4 (in=128, out=128, kernel=3, dilation=8)
        ↓
    Global Average Pooling ⊕ Last Timestep
        ↓
    FC Layer (256 → 128) + GELU + LayerNorm + Dropout(0.3)
        ↓
    FC Layer (128 → 1)
        ↓
    Output: logit (apply sigmoid for probability)

Design rationale:
    - Causal dilated convolutions ensure no future information leakage,
      critical for clinical time-series where predictions must only use
      past and current observations.
    - Exponentially increasing dilation (1, 2, 4, 8) gives a receptive
      field of 30 timesteps with kernel_size=3, covering most of the
      48-hour observation window efficiently.
    - Weight normalization stabilizes training and is preferred over
      batch normalization for sequence models.
    - Residual connections in each temporal block prevent gradient
      degradation in deeper networks.
    - TCNs offer parallelizable training (unlike sequential RNNs) and
      have shown competitive or superior performance on many sequence
      benchmarks.

References:
    - Bai et al., "An Empirical Evaluation of Generic Convolutional and
      Recurrent Networks for Sequence Modeling," 2018.

Authors: Pavana M.S., Megharaja S., Srujan S. Sagar, Abubakkar Siddiq
Guide: Prof. Shilpa K.C.
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from pathlib import Path
from typing import Tuple, Optional, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution that ensures no future information leakage.

    Applies left-side padding equal to (kernel_size - 1) * dilation
    so that the output at time t depends only on inputs at times <= t.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        dilation: Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, channels, seq_len).

        Returns:
            Output tensor, shape (batch, out_channels, seq_len).
            The output length matches the input length (causal padding
            is trimmed from the right).
        """
        out = self.conv(x)
        # Trim the right side to maintain causal property
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    A single TCN residual block with two causal convolution layers.

    Each block applies:
        x → CausalConv → ReLU → Dropout → CausalConv → ReLU → Dropout
        + residual connection (with optional 1×1 projection)

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor for this block.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.3
    ):
        super().__init__()

        # First causal convolution
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal convolution
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection: 1×1 conv if channel dimensions differ
        if in_channels != out_channels:
            self.downsample = weight_norm(nn.Conv1d(
                in_channels, out_channels, 1
            ))
        else:
            self.downsample = None

        # Final activation after residual addition
        self.relu_out = nn.ReLU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights."""
        nn.init.kaiming_normal_(self.conv1.conv.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.conv.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, in_channels, seq_len).

        Returns:
            Output tensor, shape (batch, out_channels, seq_len).
        """
        # Main path
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))

        # Residual connection
        residual = self.downsample(x) if self.downsample is not None else x

        return self.relu_out(out + residual)


class TCNSepsisModel(nn.Module):
    """
    Temporal Convolutional Network for early sepsis prediction.

    Processes hourly clinical time series data through a stack of
    causal dilated convolution blocks with exponentially increasing
    dilation factors, then aggregates the temporal representation
    for binary classification.

    Args:
        input_dim: Number of input features per timestep.
        num_channels: List of output channel sizes for each TCN block.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
        fc_hidden_dim: Hidden dimension of the classification head.
    """

    def __init__(
        self,
        input_dim: int = None,
        num_channels: List[int] = None,
        kernel_size: int = None,
        dropout: float = None,
        fc_hidden_dim: int = None
    ):
        super().__init__()

        # Use config defaults if not specified
        tc = config.TCNConfig
        self.input_dim = input_dim or tc.input_dim
        self.num_channels = num_channels or tc.num_channels
        self.kernel_size = kernel_size or tc.kernel_size
        self.dropout_rate = dropout or tc.dropout
        self.fc_hidden_dim = fc_hidden_dim or tc.fc_hidden_dim

        # ---- TCN Backbone ----
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            in_ch = self.input_dim if i == 0 else self.num_channels[i - 1]
            out_ch = self.num_channels[i]
            layers.append(TemporalBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout=self.dropout_rate
            ))

        self.tcn = nn.Sequential(*layers)

        # ---- Classification Head ----
        # Combine global average pool + last timestep → 2 * last_channel
        last_channel = self.num_channels[-1]
        classifier_input_dim = last_channel * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.fc_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.fc_hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_hidden_dim, 1)
        )

        # Store intermediate activations for interpretability
        self._feature_maps = None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_dim).
            return_attention: If True, store feature maps for later
                              retrieval (for interpretability).
                              Note: TCN does not have attention weights;
                              this flag controls feature map storage for
                              interface compatibility.

        Returns:
            Logits tensor, shape (batch, 1). Apply sigmoid for probability.
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)

        # ---- TCN Backbone ----
        tcn_out = self.tcn(x)  # (batch, last_channel, seq_len)

        if return_attention:
            self._feature_maps = tcn_out.detach()

        # ---- Pooling ----
        # Global average pooling over time
        avg_pool = tcn_out.mean(dim=2)  # (batch, last_channel)

        # Last timestep (most recent observation)
        last_step = tcn_out[:, :, -1]  # (batch, last_channel)

        # Concatenate
        combined = torch.cat([avg_pool, last_step], dim=1)  # (batch, last_channel*2)

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
        TCN does not have explicit attention weights.

        Returns None for interface compatibility. Use gradient-based
        methods (e.g., integrated gradients) for TCN interpretability.

        Returns:
            None (TCN has no attention mechanism).
        """
        return None

    def get_feature_maps(self) -> Optional[torch.Tensor]:
        """
        Retrieve stored feature maps from the last forward pass.
        Must call forward() with return_attention=True first.

        Returns:
            Feature maps, shape (batch, last_channel, seq_len),
            or None if not stored.
        """
        return self._feature_maps

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters by component.

        Returns:
            Dictionary mapping component name to parameter count.
        """
        counts = {
            "tcn_backbone": sum(
                p.numel() for n, p in self.named_parameters()
                if "tcn" in n
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

        # Compute receptive field
        receptive_field = 1
        for i in range(len(self.num_channels)):
            dilation = 2 ** i
            receptive_field += 2 * (self.kernel_size - 1) * dilation

        lines = [
            "=" * 60,
            "Temporal Convolutional Network — Sepsis Prediction Model",
            "=" * 60,
            f"  Input dim:          {self.input_dim}",
            f"  TCN channels:       {self.num_channels}",
            f"  Kernel size:        {self.kernel_size}",
            f"  Num blocks:         {len(self.num_channels)}",
            f"  Receptive field:    {receptive_field} timesteps",
            f"  FC hidden dim:      {self.fc_hidden_dim}",
            f"  Dropout:            {self.dropout_rate}",
            "-" * 60,
            f"  TCN backbone params: {params['tcn_backbone']:,}",
            f"  Classifier params:   {params['classifier']:,}",
            "-" * 60,
            f"  Total params:        {params['total']:,}",
            f"  Trainable params:    {params['trainable']:,}",
            "=" * 60,
        ]
        return "\n".join(lines)
