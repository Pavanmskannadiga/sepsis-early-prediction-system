"""
Main Training Script for the Sepsis Prediction System.

Entry point for training the BiLSTM+Attention model on the PhysioNet/CinC
2019 Challenge dataset. Supports single-hospital and combined training,
with optional SMOTE oversampling.

Usage:
    # Train on Hospital A with focal loss
    python scripts/train.py --hospital A

    # Train on Hospital A with SMOTE
    python scripts/train.py --hospital A --smote

    # Train on both hospitals combined
    python scripts/train.py --hospital AB --epochs 50

    # Quick debug run with subset
    python scripts/train.py --hospital A --subset 100 --epochs 5

    # Train baselines for comparison
    python scripts/train.py --hospital A --baselines
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.data.preprocessing import prepare_dataset
from src.data.dataset import get_dataloaders
from src.data.imbalance import apply_smote, get_loss_function
from src.models.bilstm_attention import build_model
from src.models.baselines import train_and_evaluate_baselines
from src.training.trainer import SepsisTrainer
from src.training.metrics import format_metrics_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train sepsis prediction model (BiLSTM+Attention / Transformer / TCN)"
    )
    parser.add_argument(
        "--model", type=str, default="bilstm_attention",
        choices=["bilstm_attention", "transformer", "tcn", "bilstm_tcn"],
        help="Model architecture to train"
    )
    parser.add_argument(
        "--hospital", type=str, default="A", choices=["A", "B", "AB"],
        help="Hospital data to train on (A, B, or AB for both)"
    )
    parser.add_argument(
        "--smote", action="store_true",
        help="Apply SMOTE oversampling to training data"
    )
    parser.add_argument(
        "--loss", type=str, default="focal", choices=["bce", "weighted_bce", "focal"],
        help="Loss function type"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Maximum training epochs (default from config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (default from config)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None,
        help="LSTM hidden dimension per direction"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit to N patients (for quick debugging)"
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Also train and evaluate baseline models"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cuda, or cpu"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    import time
    start_time = time.time()
    args = parse_args()
    set_seed(args.seed)

    model_names = {
        "bilstm_attention": "BiLSTM + Multi-Head Attention",
        "transformer": "Transformer Encoder",
        "tcn": "Temporal Convolutional Network (TCN)",
    }
    model_display = model_names.get(args.model, args.model)

    print("=" * 65)
    print("  Sepsis Early Prediction System — Training Pipeline")
    print(f"  Model: {model_display}")
    print("=" * 65)
    print(f"  Hospital: {args.hospital}")
    print(f"  SMOTE: {'Yes' if args.smote else 'No'}")
    print(f"  Loss: {args.loss}")
    print(f"  Device: {args.device}")
    if args.subset:
        print(f"  Subset: {args.subset} patients (debug mode)")
    print("=" * 65)

    # ---- Step 1: Load and preprocess data ----
    print("\n[Step 1/4] Loading and preprocessing data...")

    data = prepare_dataset(
        hospital=args.hospital,
        max_patients=args.subset,
        normalize=True,
        verbose=True
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # ---- Step 2: Handle class imbalance ----
    print("\n[Step 2/4] Handling class imbalance...")

    if args.smote:
        X_train, y_train = apply_smote(X_train, y_train)
        # Update data dict for DataLoader creation
        data["X_train"] = X_train
        data["y_train"] = y_train

    # Create DataLoaders
    batch_size = args.batch_size or config.TrainConfig.batch_size
    loaders = get_dataloaders(
        data,
        batch_size=batch_size,
        use_weighted_sampler=not args.smote  # Use sampler if not using SMOTE
    )

    # ---- Step 3: Build and train model ----
    print(f"\n[Step 3/4] Building and training {model_display} model...")

    model_kwargs = {}
    if args.hidden_dim:
        model_kwargs["hidden_dim"] = args.hidden_dim

    model = build_model(args.model, **model_kwargs)

    # Loss function
    criterion = get_loss_function(
        loss_type=args.loss,
        y_train=y_train if args.loss == "weighted_bce" else None
    )

    # Optimizer with optional LR override
    lr = args.lr or config.TrainConfig.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=config.TrainConfig.weight_decay
    )

    # Train
    trainer = SepsisTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device
    )

    max_epochs = args.epochs or config.TrainConfig.max_epochs
    history = trainer.train(
        loaders["train"],
        loaders["val"],
        max_epochs=max_epochs
    )

    # ---- Step 4: Evaluate ----
    print("\n[Step 4/4] Evaluating on test set...")

    # Load best model
    best_path = config.MODELS_DIR / "best_model.pt"
    if best_path.exists():
        trainer.load_checkpoint(str(best_path))

    test_metrics = trainer.evaluate(loaders["test"])

    # ---- Optional: Train baselines ----
    baseline_results = None
    if args.baselines:
        print("\nTraining baseline models for comparison...")
        baseline_results = train_and_evaluate_baselines(
            X_train, y_train, X_test, y_test
        )

    # ---- Save results ----
    results = {
        "args": vars(args),
        "test_metrics": test_metrics,
        "best_val_auroc": trainer.best_val_auroc,
        "best_epoch": trainer.best_epoch,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }

    if baseline_results:
        results["baseline_results"] = baseline_results

    results_path = config.RESULTS_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[OK] Results saved to: {results_path}")
    print(f"[OK] Best model saved to: {config.MODELS_DIR / 'best_model.pt'}")

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    
    end_time = time.time()
    training_duration = end_time - start_time
    hours, rem = divmod(training_duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"  Training Time:         {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    
    # Save training time to results
    results["training_time_seconds"] = training_duration
    print(f"  Best Validation AUROC: {trainer.best_val_auroc:.4f}")
    print(f"  Test AUROC:            {test_metrics.get('auroc', 0):.4f}")
    print(f"  Test F1:               {test_metrics.get('f1', 0):.4f}")
    print(f"  Test Sensitivity:      {test_metrics.get('sensitivity', 0):.4f}")
    print(f"  Test Specificity:      {test_metrics.get('specificity', 0):.4f}")
    print("=" * 65)

    if test_metrics.get("auroc", 0) >= 0.90:
        print("  [TARGET MET] AUROC >= 0.90")
    else:
        print(f"  [WARNING] Below target: AUROC = {test_metrics.get('auroc', 0):.4f} (target: 0.90)")

    return results


if __name__ == "__main__":
    main()
