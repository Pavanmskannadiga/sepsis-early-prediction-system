"""
Class Imbalance Handling Module.

Provides SMOTE, ADASYN, SMOTE-ENN, focal loss, and class-weighted loss
implementations for handling the severe sepsis class imbalance (~8% positive).

IMPORTANT: SMOTE is applied ONLY to training data, NEVER to validation/test.
For time-series data, SMOTE is applied on flattened sequences and reshaped
back. This may break fine temporal correlations, so we also provide
alternatives: focal loss, weighted BCE, and WeightedRandomSampler.

References:
    - Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," 2002
    - Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


# ============================================================
# SMOTE-based Oversampling
# ============================================================

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "auto",
    k_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to training sequences.

    For 3D time-series data (samples, timesteps, features), sequences are
    flattened to 2D for SMOTE and reshaped back. This is the most common
    approach in the sepsis-prediction literature.

    Args:
        X_train: Training features, shape (n, T, F).
        y_train: Training labels, shape (n,).
        strategy: SMOTE sampling strategy.
        k_neighbors: Number of nearest neighbors for SMOTE.
        random_state: Random seed for reproducibility.

    Returns:
        Resampled (X_train, y_train).
    """
    from imblearn.over_sampling import SMOTE

    original_shape = X_train.shape
    is_3d = X_train.ndim == 3

    if is_3d:
        n_samples, seq_len, n_features = X_train.shape
        X_flat = X_train.reshape(n_samples, -1)
    else:
        X_flat = X_train

    print(f"Before SMOTE: {np.bincount(y_train.astype(int))}")

    smote = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    X_resampled, y_resampled = smote.fit_resample(X_flat, y_train)

    print(f"After SMOTE:  {np.bincount(y_resampled.astype(int))}")

    if is_3d:
        X_resampled = X_resampled.reshape(-1, seq_len, n_features)

    return X_resampled.astype(np.float32), y_resampled.astype(np.int32)


def apply_adasyn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ADASYN (Adaptive Synthetic Sampling) oversampling.
    ADASYN generates more synthetic samples for minority instances
    that are harder to learn.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_neighbors: Number of nearest neighbors.
        random_state: Random seed.

    Returns:
        Resampled (X_train, y_train).
    """
    from imblearn.over_sampling import ADASYN

    is_3d = X_train.ndim == 3
    if is_3d:
        n_samples, seq_len, n_features = X_train.shape
        X_flat = X_train.reshape(n_samples, -1)
    else:
        X_flat = X_train

    print(f"Before ADASYN: {np.bincount(y_train.astype(int))}")

    adasyn = ADASYN(
        n_neighbors=n_neighbors,
        random_state=random_state
    )
    X_resampled, y_resampled = adasyn.fit_resample(X_flat, y_train)

    print(f"After ADASYN:  {np.bincount(y_resampled.astype(int))}")

    if is_3d:
        X_resampled = X_resampled.reshape(-1, seq_len, n_features)

    return X_resampled.astype(np.float32), y_resampled.astype(np.int32)


def apply_smote_enn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE + Edited Nearest Neighbors (SMOTE-ENN) hybrid.
    First oversamples with SMOTE, then cleans the decision boundary
    by removing samples that are misclassified by their neighbors.

    This can improve precision by removing ambiguous synthetic samples,
    but may discard real borderline patients.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed.

    Returns:
        Resampled and cleaned (X_train, y_train).
    """
    from imblearn.combine import SMOTEENN

    is_3d = X_train.ndim == 3
    if is_3d:
        n_samples, seq_len, n_features = X_train.shape
        X_flat = X_train.reshape(n_samples, -1)
    else:
        X_flat = X_train

    print(f"Before SMOTE-ENN: {np.bincount(y_train.astype(int))}")

    smote_enn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X_flat, y_train)

    print(f"After SMOTE-ENN:  {np.bincount(y_resampled.astype(int))}")

    if is_3d:
        X_resampled = X_resampled.reshape(-1, seq_len, n_features)

    return X_resampled.astype(np.float32), y_resampled.astype(np.int32)


# ============================================================
# Loss Functions for Imbalance
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Down-weights well-classified (easy) examples and focuses training
    on hard, misclassified examples. Particularly effective for extreme
    imbalance ratios like the ~8% sepsis prevalence.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples).
               Default=2.0 as per Lin et al., 2017.
        alpha: Balance weight for positive class. Default=0.75.
        reduction: "mean" or "sum".

    References:
        Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw model outputs (logits), shape (batch,).
            targets: Ground truth labels (0 or 1), shape (batch,).

        Returns:
            Focal loss value.
        """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Apply focal weights
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-frequency class weight for the positive class,
    suitable for use as pos_weight in BCEWithLogitsLoss.

    Args:
        y_train: Training labels array.

    Returns:
        pos_weight tensor for BCEWithLogitsLoss.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        return torch.tensor([1.0])
    pos_weight = n_neg / n_pos
    pos_weight *= config.ImbalanceConfig.pos_weight_multiplier
    print(f"Class weights: neg={n_neg}, pos={n_pos}, pos_weight={pos_weight:.2f}")
    return torch.tensor([pos_weight], dtype=torch.float32)


def get_loss_function(
    loss_type: str = "focal",
    y_train: Optional[np.ndarray] = None,
    device: str = "cpu"
) -> nn.Module:
    """
    Factory function to create the appropriate loss function.

    Args:
        loss_type: "bce", "weighted_bce", or "focal".
        y_train: Training labels (needed for weighted_bce).
        device: Target device.

    Returns:
        PyTorch loss module.
    """
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    elif loss_type == "weighted_bce":
        if y_train is None:
            raise ValueError("y_train required for weighted BCE loss")
        pos_weight = get_class_weights(y_train).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    elif loss_type == "focal":
        return FocalLoss(
            gamma=config.TrainConfig.focal_gamma,
            alpha=config.TrainConfig.focal_alpha
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
