"""
Training Pipeline for the Sepsis Prediction System.

Implements the complete training loop with:
  - Early stopping on validation AUROC
  - Learning rate scheduling (cosine annealing)
  - Gradient clipping for LSTM stability
  - Model checkpointing (best validation performance)
  - Training/validation loss and metric logging
  - Support for focal loss and weighted BCE for imbalance

Authors: Pavana M.S., Megharaja S., Srujan S. Sagar, Abubakkar Siddiq
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.training.metrics import compute_all_metrics, find_optimal_threshold, format_metrics_table


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.

    Monitors a specified metric (e.g., val_auroc) and stops training
    after 'patience' epochs without improvement beyond 'min_delta'.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class SepsisTrainer:
    """
    Training manager for the BiLSTM sepsis prediction model.

    Handles the complete training lifecycle: forward/backward passes,
    validation, early stopping, learning rate scheduling, gradient
    clipping, model checkpointing, and metric logging.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        device: str = "auto",
        checkpoint_dir: str = None
    ):
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device) if hasattr(criterion, 'to') else criterion

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.TrainConfig.learning_rate,
                weight_decay=config.TrainConfig.weight_decay
            )
        else:
            self.optimizer = optimizer

        # Learning rate scheduler
        if scheduler is None:
            if config.TrainConfig.lr_scheduler == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.TrainConfig.max_epochs,
                    eta_min=config.TrainConfig.lr_min
                )
            elif config.TrainConfig.lr_scheduler == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config.TrainConfig.lr_step_size,
                    gamma=config.TrainConfig.lr_gamma
                )
            elif config.TrainConfig.lr_scheduler == "plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="max",
                    factor=0.5,
                    patience=5
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = scheduler

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir or config.MODELS_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.TrainConfig.patience,
            min_delta=config.TrainConfig.min_delta,
            mode="max"
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "val_auprc": [],
            "val_f1": [],
            "learning_rate": [],
        }

        self.best_val_auroc = 0.0
        self.best_epoch = 0

        # Infer model type
        class_name = self.model.__class__.__name__
        if "BiLSTMTCN" in class_name:
            self.model_type = "bilstm_tcn"
        elif "Transformer" in class_name:
            self.model_type = "transformer"
        elif "TCN" in class_name:
            self.model_type = "tcn"
        elif "SimpleLSTM" in class_name:
            self.model_type = "simple_lstm"
        else:
            self.model_type = "bilstm_attention"

        print(f"Trainer initialized:")
        print(f"  Model type: {self.model_type}")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Optimizer: {type(self.optimizer).__name__}")
        print(f"  Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Run one training epoch.

        Args:
            train_loader: Training DataLoader.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_x).squeeze(-1)
            loss = self.criterion(logits, batch_y)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.TrainConfig.gradient_clip_norm
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Run validation and compute metrics.

        Args:
            val_loader: Validation DataLoader.

        Returns:
            Tuple of (val_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []
        n_batches = 0

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits = self.model(batch_x).squeeze(-1)
            loss = self.criterion(logits, batch_y)
            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)

        metrics = compute_all_metrics(y_true, y_prob)

        return avg_loss, metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = None,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Full training loop with early stopping and checkpointing.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            max_epochs: Maximum training epochs.
            verbose: Show progress.

        Returns:
            Training history dictionary.
        """
        max_epochs = max_epochs or config.TrainConfig.max_epochs

        if verbose:
            print(f"\nStarting training for up to {max_epochs} epochs...")
            print(f"{'='*75}")
            print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} "
                  f"{'AUROC':>7} {'AUPRC':>7} {'F1':>6} {'LR':>10}")
            print(f"{'='*75}")

        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auroc"].append(val_metrics["auroc"])
            self.history["val_auprc"].append(val_metrics["auprc"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["learning_rate"].append(current_lr)

            # Check for best model
            is_best = val_metrics["auroc"] > self.best_val_auroc
            if is_best:
                self.best_val_auroc = val_metrics["auroc"]
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_metrics, is_best=True)

            # Print progress
            if verbose:
                best_marker = " *" if is_best else ""
                print(f"{epoch:>6d} {train_loss:>11.4f} {val_loss:>10.4f} "
                      f"{val_metrics['auroc']:>7.4f} {val_metrics['auprc']:>7.4f} "
                      f"{val_metrics['f1']:>6.3f} {current_lr:>10.2e}{best_marker}")

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["auroc"])
                else:
                    self.scheduler.step()

            # Early stopping check
            if self.early_stopping(val_metrics["auroc"]):
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best epoch: {self.best_epoch} with AUROC={self.best_val_auroc:.4f}")
                break

        total_time = time.time() - start_time

        if verbose:
            print(f"{'='*75}")
            print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"Best validation AUROC: {self.best_val_auroc:.4f} at epoch {self.best_epoch}")

        return self.history

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_type": self.model_type,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_val_auroc": self.best_val_auroc,
            "history": self.history,
            "config": {
                "model": {k: v for k, v in config.ModelConfig.__dict__.items() if not k.startswith("__")},
                "train": {k: v for k, v in config.TrainConfig.__dict__.items() if not k.startswith("__")},
            }
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save latest
        path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load a saved checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        if "metrics" in checkpoint:
            print(f"  AUROC: {checkpoint['metrics'].get('auroc', '?'):.4f}")

        return checkpoint

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        find_best_threshold: bool = True,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Final evaluation on test set.

        Args:
            test_loader: Test DataLoader.
            find_best_threshold: If True, find optimal threshold on test data.
            verbose: Print results.

        Returns:
            Metrics dictionary.
        """
        self.model.eval()
        all_probs = []
        all_labels = []

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(self.device)
            logits = self.model(batch_x).squeeze(-1)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch_y.numpy())

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)

        # Metrics at default threshold
        metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)

        # Find optimal threshold
        if find_best_threshold:
            opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_prob, metric="f1")
            metrics_optimal = compute_all_metrics(y_true, y_prob, threshold=opt_thresh)
            metrics["optimal_threshold"] = opt_thresh
            metrics["optimal_f1"] = opt_f1
            metrics["optimal_sensitivity"] = metrics_optimal["sensitivity"]
            metrics["optimal_specificity"] = metrics_optimal["specificity"]

        if verbose:
            print(format_metrics_table(metrics, "Test Set Evaluation"))
            if find_best_threshold:
                print(f"\n  Optimal threshold: {opt_thresh:.2f}")
                print(f"  F1 at optimal:     {opt_f1:.4f}")

        return metrics

    def get_predictions(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for a dataset.

        Returns:
            Tuple of (y_true, y_prob) arrays.
        """
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x).squeeze(-1)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch_y.numpy())

        return np.concatenate(all_labels), np.concatenate(all_probs)
