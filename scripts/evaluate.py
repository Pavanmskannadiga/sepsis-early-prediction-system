"""
Cross-Hospital Evaluation Script.

Runs cross-hospital validation experiments to test generalizability
(Research Objective O3). Also supports ablation studies.

Usage:
    # Run all cross-hospital experiments
    python scripts/evaluate.py --cross-hospital

    # Quick evaluation with subset
    python scripts/evaluate.py --cross-hospital --subset 100 --epochs 10

    # Run ablation study
    python scripts/evaluate.py --ablation --hospital A --epochs 20

    # Evaluate a specific trained model
    python scripts/evaluate.py --model results/models/best_model.pt --hospital B
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
from src.models.bilstm_attention import BiLSTMAttention
from src.training.trainer import SepsisTrainer
from src.training.metrics import compute_all_metrics, format_metrics_table
from src.evaluation.cross_hospital import (
    run_cross_hospital_experiment,
    run_all_experiments,
    generate_comparison_table
)
from src.data.imbalance import get_loss_function


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate sepsis prediction model"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model checkpoint (for single evaluation)"
    )
    parser.add_argument(
        "--hospital", type=str, default="A", choices=["A", "B"],
        help="Test hospital (for single evaluation or ablation)"
    )
    parser.add_argument(
        "--cross-hospital", action="store_true",
        help="Run all cross-hospital experiments"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study (compares model components)"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit patients per hospital"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Max epochs for cross-hospital/ablation training"
    )
    return parser.parse_args()


def evaluate_single_model(model_path: str, hospital: str, subset: int = None):
    """Evaluate a pre-trained model on a hospital's test set."""

    print("=" * 65)
    print(f"  Evaluating model on Hospital {hospital}")
    print("=" * 65)

    # Load data
    data = prepare_dataset(hospital=hospital, max_patients=subset)

    # Load model
    model = BiLSTMAttention()
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")

    # Create test loader
    loaders = get_dataloaders(data, batch_size=config.TrainConfig.batch_size)

    # Evaluate
    criterion = get_loss_function("focal")
    trainer = SepsisTrainer(model=model, criterion=criterion)
    metrics = trainer.evaluate(loaders["test"])

    return metrics


def main():
    args = parse_args()

    if args.ablation:
        print("=" * 65)
        print("  Ablation Study — Component Contribution Analysis")
        print("=" * 65)
        try:
            from src.evaluation.ablation import run_ablation_study, format_ablation_table, save_ablation_results
        except ImportError as e:
            print(f"[ERROR] Cannot import ablation module: {e}")
            return

        max_epochs = args.epochs or 20
        results = run_ablation_study(
            hospital=args.hospital,
            max_patients=args.subset,
            max_epochs=max_epochs,
            verbose=True
        )
        table = format_ablation_table(results)
        print("\n" + table)
        save_ablation_results(results)
        print(f"\n[OK] Ablation results saved to: {config.TABLES_DIR / 'ablation_results.csv'}")

    elif args.cross_hospital:
        print("=" * 65)
        print("  Cross-Hospital Validation - Full Experiment Suite")
        print("  (Research Objective O3: Cross-Hospital Generalization)")
        print("=" * 65)

        results = run_all_experiments(
            max_patients=args.subset,
            max_epochs=args.epochs
        )

        # Generate comparison table
        table = generate_comparison_table(
            results,
            save_path=str(config.TABLES_DIR / "cross_hospital_results.csv")
        )

        # Save full results
        results_path = config.RESULTS_DIR / "cross_hospital_full_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n[OK] Full results saved to: {results_path}")

    elif args.model:
        metrics = evaluate_single_model(
            args.model, args.hospital, args.subset
        )

    else:
        print("Usage:")
        print("  python scripts/evaluate.py --cross-hospital")
        print("  python scripts/evaluate.py --ablation --hospital A --epochs 20")
        print("  python scripts/evaluate.py --model path/to/model.pt --hospital B")


if __name__ == "__main__":
    main()
