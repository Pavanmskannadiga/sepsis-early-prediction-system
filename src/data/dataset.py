"""
PyTorch Dataset and DataLoader module for the Sepsis Prediction System.

Wraps preprocessed numpy arrays into PyTorch Dataset objects and provides
factory functions for creating DataLoaders with proper batching, shuffling,
and optional weighted sampling for class imbalance.
"""

import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Tuple, Optional, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


class SepsisDataset(Dataset):
    """
    PyTorch Dataset for sepsis prediction sequences.

    Each sample is a tuple of:
      - x: Feature sequence tensor, shape (seq_len, n_features)
      - y: Binary label (0=no sepsis, 1=sepsis within 6 hours)

    The dataset supports optional sample weighting for handling
    class imbalance via WeightedRandomSampler.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Feature sequences, shape (n_samples, seq_len, n_features).
            y: Labels, shape (n_samples,).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def n_positive(self) -> int:
        """Number of positive (sepsis) samples."""
        return int(self.y.sum().item())

    @property
    def n_negative(self) -> int:
        """Number of negative (no sepsis) samples."""
        return len(self.y) - self.n_positive

    @property
    def pos_weight(self) -> float:
        """Inverse frequency weight for positive class (n_neg / n_pos)."""
        if self.n_positive == 0:
            return 1.0
        return self.n_negative / self.n_positive

    @property
    def class_distribution(self) -> Dict[str, float]:
        """Return class distribution as percentages."""
        n = len(self.y)
        return {
            "positive": 100.0 * self.n_positive / n,
            "negative": 100.0 * self.n_negative / n,
            "ratio": f"1:{self.n_negative // max(1, self.n_positive)}"
        }


def create_weighted_sampler(dataset: SepsisDataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that oversamples the minority class
    (sepsis) during training. This is an alternative to SMOTE that
    preserves temporal integrity by not creating synthetic sequences.

    The sampler assigns a weight inversely proportional to class frequency
    so that each batch has approximately equal representation.

    Args:
        dataset: A SepsisDataset instance.

    Returns:
        WeightedRandomSampler for use with DataLoader.
    """
    labels = dataset.y.numpy()
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def get_dataloaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int = 64,
    use_weighted_sampler: bool = False,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from preprocessed data.

    Args:
        data_dict: Dictionary with X_train, y_train, X_val, y_val,
                   X_test, y_test arrays (from preprocessing.prepare_dataset).
        batch_size: Batch size.
        use_weighted_sampler: If True, use WeightedRandomSampler for
                              training to handle class imbalance.
        num_workers: Number of data loading workers.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoader objects.
    """
    train_dataset = SepsisDataset(data_dict["X_train"], data_dict["y_train"])
    val_dataset = SepsisDataset(data_dict["X_val"], data_dict["y_val"])
    test_dataset = SepsisDataset(data_dict["X_test"], data_dict["y_test"])

    # Training DataLoader — with optional weighted sampler
    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    # Validation DataLoader — no shuffling, no sampling
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Test DataLoader — no shuffling, no sampling
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"    Distribution: {train_dataset.class_distribution}")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Weighted sampler: {'Yes' if use_weighted_sampler else 'No'}")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }
