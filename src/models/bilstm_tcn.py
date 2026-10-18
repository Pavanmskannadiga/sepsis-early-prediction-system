"""
BiLSTM + TCN Hybrid Model for Sepsis Early Prediction.

This hybrid model combines Temporal Convolutional Network (TCN) blocks
with a Bidirectional LSTM (BiLSTM).

Architecture:
    Input (batch, seq_len=48, features=68)
        ↓
    TCN Temporal Blocks (Dilations: 1, 2, 4)
      - Extracts multi-scale localized causal features and vital dynamics
        ↓
    BiLSTM Layer (hidden=128, bidirectional → 256)
      - Captures long-range temporal dependencies and global trajectory
        ↓
    Dual Temporal Pooling (Last Hidden State ⊕ Global Max Pooling → 512)
        ↓
    FC Layer (512 → 128) + GELU + LayerNorm + Dropout(0.3)
        ↓
    FC Layer (128 → 1)
        ↓
    Output: logit (apply sigmoid for probability)
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
from src.models.tcn_model import TemporalBlock


class BiLSTMTCN(nn.Module):
    """
    Hybrid BiLSTM + TCN model for early sepsis prediction.
    """

    def __init__(
        self,
        input_dim: int = None,
        tcn_channels: int = 128,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 1,
        dropout: float = 0.3,
        fc_hidden_dim: int = 128
    ):
        super().__init__()
        mc = config.ModelConfig
        self.input_dim = input_dim or mc.input_dim
        self.tcn_channels = tcn_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout
        self.fc_hidden_dim = fc_hidden_dim

        # ---- TCN Feature Extractor ----
        # 3 residual blocks with dilations 1, 2, 4
        self.tcn_blocks = nn.ModuleList([
            TemporalBlock(
                in_channels=self.input_dim,
                out_channels=self.tcn_channels,
                kernel_size=3,
                dilation=1,
                dropout=self.dropout_rate
            ),
            TemporalBlock(
                in_channels=self.tcn_channels,
                out_channels=self.tcn_channels,
                kernel_size=3,
                dilation=2,
                dropout=self.dropout_rate
            ),
            TemporalBlock(
                in_channels=self.tcn_channels,
                out_channels=self.tcn_channels,
                kernel_size=3,
                dilation=4,
                dropout=self.dropout_rate
            ),
        ])

        # ---- BiLSTM Recurrent Layer ----
        self.lstm = nn.LSTM(
            input_size=self.tcn_channels,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_lstm_layers > 1 else 0.0
        )
        self.lstm_output_dim = self.lstm_hidden_dim * 2
        self.layer_norm = nn.LayerNorm(self.lstm_output_dim)

        # ---- Classification Head ----
        # Combining last timestep (256) + global max pool (256) = 512
        classifier_input_dim = self.lstm_output_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.fc_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.fc_hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers and LSTM bias setup."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor, return_attention: bool = False, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_dim).
            return_attention: Dummy parameter for API compatibility.

        Returns:
            Logits tensor, shape (batch, 1).
        """
        # x is (batch, seq_len, input_dim) -> transpose to (batch, input_dim, seq_len) for TCN
        tcn_out = x.transpose(1, 2)
        for block in self.tcn_blocks:
            tcn_out = block(tcn_out)

        # Transpose back: (batch, tcn_channels, seq_len) -> (batch, seq_len, tcn_channels)
        lstm_in = tcn_out.transpose(1, 2)

        # BiLSTM forward
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.layer_norm(lstm_out)

        # Dual pooling: Last timestep + Global max pooling over sequence
        last_step = lstm_out[:, -1, :]
        max_pool, _ = lstm_out.max(dim=1)
        pooled = torch.cat([last_step, max_pool], dim=1)

        # Classifier
        logits = self.classifier(pooled)
        return logits

    def get_num_params(self) -> Dict[str, int]:
        tcn_params = sum(p.numel() for block in self.tcn_blocks for p in block.parameters())
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "tcn": tcn_params,
            "lstm": lstm_params,
            "classifier": classifier_params,
            "total": total,
            "trainable": trainable
        }

    def summary(self) -> str:
        params = self.get_num_params()
        return (
            f"\n{'='*60}\n"
            f"BiLSTM + TCN Hybrid Sepsis Prediction Model\n"
            f"{'='*60}\n"
            f"  Input features:       {self.input_dim}\n"
            f"  TCN channels:         {self.tcn_channels} (3 blocks, d=1,2,4)\n"
            f"  BiLSTM hidden dim:    {self.lstm_hidden_dim} (x2 bidirectional = {self.lstm_output_dim})\n"
            f"  Classifier input dim: {self.lstm_output_dim * 2}\n"
            f"  Classifier hidden:    {self.fc_hidden_dim}\n"
            f"  Dropout:              {self.dropout_rate}\n"
            f"{'-'*60}\n"
            f"  TCN parameters:       {params['tcn']:,}\n"
            f"  BiLSTM parameters:    {params['lstm']:,}\n"
            f"  Classifier params:    {params['classifier']:,}\n"
            f"{'-'*60}\n"
            f"  Total parameters:     {params['total']:,}\n"
            f"  Trainable parameters: {params['trainable']:,}\n"
            f"{'='*60}\n"
        )
