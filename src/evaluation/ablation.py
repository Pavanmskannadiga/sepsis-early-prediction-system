"""
Ablation Study Module.

Evaluates the contribution of different model components (ablation study)
to justify design choices for the explainable BiLSTM sepsis prediction system.

Ablations:
  1. Full Model (Proposed: BiLSTM + Multi-Head Self-Attention + Masks + Focal Loss)
  2. Ablating Self-Attention (BiLSTM with simple global max pooling, no attention)
  3. Ablating Bidirectionality (Unidirectional LSTM + Attention)
  4. Ablating Missingness Masks (Input features without binary missingness masks)
  5. Ablating Class Imbalance Handling (Standard BCE loss on raw distribution, no Focal Loss/Sampler)
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.data.preprocessing import prepare_dataset, normalize_features
from src.data.dataset import get_dataloaders
from src.data.imbalance import get_loss_function
from src.models.bilstm_attention import BiLSTMAttention, SimpleLSTM, build_model
from src.training.trainer import SepsisTrainer
from src.training.metrics import compute_all_metrics


class BiLSTMNoAttention(nn.Module):
    """Ablated model: BiLSTM with standard max pooling, no attention mechanism."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        # Global max pooling over time (ablatting attention weights)
        pooled, _ = lstm_out.max(dim=1)
        return self.fc(pooled)


class UnidirectionalLSTMAttention(nn.Module):
    """Ablated model: Unidirectional LSTM with Multi-Head Attention."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        from src.models.bilstm_attention import MultiHeadAttention
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(embed_dim=hidden_dim, num_heads=4, dropout=dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        attn_out, _ = self.attention(lstm_out)
        attn_out = self.attention_norm(attn_out + lstm_out)
        
        context = attn_out.mean(dim=1)
        max_pool, _ = lstm_out.max(dim=1)
        combined = torch.cat([context, max_pool], dim=1)
        return self.classifier(combined)


def run_ablation_study(
    hospital: str = "A",
    max_patients: Optional[int] = None,
    max_epochs: int = 10,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run the full suite of ablation configurations.

    Args:
        hospital: Hospital data split to use ("A" or "B").
        max_patients: Limit patients for quick run.
        max_epochs: Maximum epochs per configuration.
        verbose: Print progress.

    Returns:
        Dict mapping configuration name to test metrics.
    """
    results = {}
    
    # Load dataset splits (with masks)
    if verbose:
        print("\nLoading dataset for ablation study...")
    data_full = prepare_dataset(hospital=hospital, max_patients=max_patients, normalize=True, verbose=verbose)
    
    # Configure each ablation run
    configs = [
        {"name": "Full Model (BiLSTM+Attn+Masks+FocalLoss)", "model_type": "full", "use_masks": True, "loss_type": "focal", "use_sampler": True},
        {"name": "Transformer Encoder", "model_type": "transformer", "use_masks": True, "loss_type": "focal", "use_sampler": True},
        {"name": "TCN (Temporal Convolutional Network)", "model_type": "tcn", "use_masks": True, "loss_type": "focal", "use_sampler": True},
        {"name": "Ablated: Self-Attention (BiLSTM only)", "model_type": "no_attn", "use_masks": True, "loss_type": "focal", "use_sampler": True},
        {"name": "Ablated: Bidirectionality (UniLSTM+Attn)", "model_type": "uni", "use_masks": True, "loss_type": "focal", "use_sampler": True},
        {"name": "Ablated: Missingness Masks (Vitals/Labs only)", "model_type": "full", "use_masks": False, "loss_type": "focal", "use_sampler": True},
        {"name": "Ablated: Imbalance Handling (Standard BCE)", "model_type": "full", "use_masks": True, "loss_type": "bce", "use_sampler": False},
    ]

    for cfg in configs:
        name = cfg["name"]
        if verbose:
            print(f"\n{'-'*65}")
            print(f" Running Ablation: {name}")
            print(f"{'-'*65}")

        # Data split setup
        if not cfg["use_masks"]:
            # Slice features to clinical only (discard masks: first 34 columns)
            dim = config.NUM_CLINICAL_FEATURES
            X_tr = data_full["X_train"][:, :, :dim]
            X_v = data_full["X_val"][:, :, :dim]
            X_te = data_full["X_test"][:, :, :dim]
        else:
            dim = config.NUM_INPUT_FEATURES
            X_tr, X_v, X_te = data_full["X_train"], data_full["X_val"], data_full["X_test"]

        loaders = get_dataloaders(
            {"X_train": X_tr, "y_train": data_full["y_train"],
             "X_val": X_v, "y_val": data_full["y_val"],
             "X_test": X_te, "y_test": data_full["y_test"]},
            batch_size=config.TrainConfig.batch_size,
            use_weighted_sampler=cfg["use_sampler"]
        )

        # Model selection
        if cfg["model_type"] == "full":
            model = BiLSTMAttention(input_dim=dim)
        elif cfg["model_type"] == "no_attn":
            model = BiLSTMNoAttention(input_dim=dim)
        elif cfg["model_type"] == "uni":
            model = UnidirectionalLSTMAttention(input_dim=dim)
        elif cfg["model_type"] == "transformer":
            model = build_model("transformer", input_dim=dim)
        elif cfg["model_type"] == "tcn":
            model = build_model("tcn", input_dim=dim)
        else:
            raise ValueError(f"Unknown model type: {cfg['model_type']}")
        # Loss selection
        criterion = get_loss_function(
            loss_type=cfg["loss_type"],
            y_train=data_full["y_train"] if cfg["loss_type"] == "weighted_bce" else None
        )

        # Train configuration
        trainer = SepsisTrainer(
            model=model,
            criterion=criterion,
            checkpoint_dir=str(config.MODELS_DIR / f"ablation_{cfg['model_type']}")
        )

        # Run training loop
        trainer.train(loaders["train"], loaders["val"], max_epochs=max_epochs, verbose=False)

        # Evaluate best epoch performance
        test_metrics = trainer.evaluate(loaders["test"], find_best_threshold=True, verbose=verbose)
        results[name] = test_metrics

    return results


def format_ablation_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format ablation metrics into a readable terminal table."""
    lines = [
        "=" * 90,
        "  Ablation Study Results",
        "=" * 90,
        f"{'Configuration':<45} {'AUROC':>7} {'AUPRC':>7} {'F1':>6} {'Sens':>6} {'Spec':>6}",
        "-" * 90,
    ]

    for name, metrics in results.items():
        lines.append(
            f"{name:<45} "
            f"{metrics.get('auroc', 0):.4f}  "
            f"{metrics.get('auprc', 0):.4f}  "
            f"{metrics.get('f1', 0):.3f}  "
            f"{metrics.get('sensitivity', 0):.3f}  "
            f"{metrics.get('specificity', 0):.3f}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


def save_ablation_results(results: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
    """Save ablation outcomes to a standard CSV report file."""
    if save_path is None:
        save_path = str(config.TABLES_DIR / "ablation_results.csv")

    rows = []
    for name, metrics in results.items():
        rows.append({
            "Configuration": name,
            "AUROC": metrics.get("auroc", 0),
            "AUPRC": metrics.get("auprc", 0),
            "F1": metrics.get("f1", 0),
            "Sensitivity": metrics.get("sensitivity", 0),
            "Specificity": metrics.get("specificity", 0),
            "Accuracy": metrics.get("accuracy", 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
