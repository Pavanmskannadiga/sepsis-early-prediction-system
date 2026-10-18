"""
Cross-Hospital Validation Module.

Implements Research Objective O3: validate the trained model across
independent hospital cohorts to test cross-hospital generalizability.

The PhysioNet/CinC 2019 dataset contains data from two hospitals
(training_setA and training_setB), enabling direct cross-hospital
transfer experiments:
  - Train on Hospital A → Test on Hospital B
  - Train on Hospital B → Test on Hospital A
  - Train on both → Test on held-out from each
  - Compare performance with and without SMOTE

This addresses the "poor cross-hospital transfer" gap (G2) identified
in Section IX of the survey paper.
"""

import sys
import os
import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.data.preprocessing import process_hospital_data, normalize_features
from src.data.dataset import SepsisDataset, get_dataloaders
from src.data.imbalance import apply_smote, get_loss_function
from src.models.bilstm_attention import BiLSTMAttention, build_model
from src.training.trainer import SepsisTrainer
from src.training.metrics import compute_all_metrics, format_metrics_table


def run_cross_hospital_experiment(
    train_hospital: str,
    test_hospital: str,
    use_smote: bool = False,
    max_patients: Optional[int] = None,
    max_epochs: int = None,
    model_type: str = "bilstm_attention",
    verbose: bool = True
) -> Dict:
    """
    Run a single cross-hospital experiment.

    Args:
        train_hospital: "A", "B", or "AB".
        test_hospital: "A" or "B".
        use_smote: Whether to apply SMOTE to training data.
        max_patients: Limit patients (for debugging).
        max_epochs: Override max epochs.
        verbose: Show progress.

    Returns:
        Dictionary with experiment configuration and results.
    """
    experiment_name = f"Train_{train_hospital}_Test_{test_hospital}"
    if use_smote:
        experiment_name += "_SMOTE"

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Cross-Hospital Experiment: {experiment_name}")
        print(f"{'='*65}")

    start_time = time.time()

    # ---- Load training data ----
    if verbose:
        print(f"\n[1/5] Loading training data (Hospital {train_hospital})...")

    if train_hospital == "AB":
        X_a, y_a, _ = process_hospital_data(
            str(config.HOSPITAL_A_DIR), max_patients, verbose=verbose
        )
        X_b, y_b, _ = process_hospital_data(
            str(config.HOSPITAL_B_DIR), max_patients, verbose=verbose
        )
        X_train_full = np.concatenate([X_a, X_b], axis=0)
        y_train_full = np.concatenate([y_a, y_b], axis=0)
    elif train_hospital == "A":
        X_train_full, y_train_full, _ = process_hospital_data(
            str(config.HOSPITAL_A_DIR), max_patients, verbose=verbose
        )
    elif train_hospital == "B":
        X_train_full, y_train_full, _ = process_hospital_data(
            str(config.HOSPITAL_B_DIR), max_patients, verbose=verbose
        )

    # ---- Load test data ----
    if verbose:
        print(f"\n[2/5] Loading test data (Hospital {test_hospital})...")

    if test_hospital == "A":
        X_test, y_test, _ = process_hospital_data(
            str(config.HOSPITAL_A_DIR), max_patients, verbose=verbose
        )
    elif test_hospital == "B":
        X_test, y_test, _ = process_hospital_data(
            str(config.HOSPITAL_B_DIR), max_patients, verbose=verbose
        )

    # ---- Split training into train/val ----
    np.random.seed(config.TrainConfig.random_seed)
    indices = np.random.permutation(len(X_train_full))
    X_train_full = X_train_full[indices]
    y_train_full = y_train_full[indices]

    n_val = int(len(X_train_full) * 0.15)
    X_val = X_train_full[:n_val]
    y_val = y_train_full[:n_val]
    X_train = X_train_full[n_val:]
    y_train = y_train_full[n_val:]

    # ---- Normalize ----
    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)

    # ---- Apply SMOTE if requested ----
    if use_smote:
        if verbose:
            print(f"\n[3/5] Applying SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train)
    else:
        if verbose:
            print(f"\n[3/5] Skipping SMOTE (using focal loss instead)")

    # ---- Create DataLoaders ----
    data_dict = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }
    loaders = get_dataloaders(data_dict, batch_size=config.TrainConfig.batch_size)

    # ---- Build and train model ----
    model_names = {
        "bilstm_attention": "BiLSTM+Attention",
        "transformer": "Transformer Encoder",
        "tcn": "TCN",
    }
    model_display = model_names.get(model_type, model_type)
    if verbose:
        print(f"\n[4/5] Training {model_display} model...")

    model = build_model(model_type)
    criterion = get_loss_function(
        loss_type="focal" if not use_smote else "bce",
        y_train=y_train
    )

    checkpoint_dir = config.MODELS_DIR / experiment_name
    trainer = SepsisTrainer(
        model=model,
        criterion=criterion,
        checkpoint_dir=str(checkpoint_dir)
    )

    history = trainer.train(
        loaders["train"],
        loaders["val"],
        max_epochs=max_epochs or config.TrainConfig.max_epochs,
        verbose=verbose
    )

    # ---- Evaluate on test set ----
    if verbose:
        print(f"\n[5/5] Evaluating on Hospital {test_hospital} test set...")

    # Load best model
    best_model_path = checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        trainer.load_checkpoint(str(best_model_path))

    test_metrics = trainer.evaluate(loaders["test"], verbose=verbose)

    elapsed = time.time() - start_time

    result = {
        "experiment_name": experiment_name,
        "train_hospital": train_hospital,
        "test_hospital": test_hospital,
        "use_smote": use_smote,
        "metrics": test_metrics,
        "best_val_auroc": trainer.best_val_auroc,
        "best_epoch": trainer.best_epoch,
        "elapsed_seconds": elapsed,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }

    return result


def run_all_experiments(
    max_patients: Optional[int] = None,
    max_epochs: int = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Run all cross-hospital experiments defined in config.

    Returns:
        List of experiment result dictionaries.
    """
    experiments = config.CrossHospitalConfig.experiments
    all_results = []

    for exp in experiments:
        train = exp["train"]
        test = exp["test"]

        # Run without SMOTE
        result = run_cross_hospital_experiment(
            train, test,
            use_smote=False,
            max_patients=max_patients,
            max_epochs=max_epochs,
            verbose=verbose
        )
        all_results.append(result)

        # Run with SMOTE if configured
        if config.CrossHospitalConfig.compare_smote:
            result_smote = run_cross_hospital_experiment(
                train, test,
                use_smote=True,
                max_patients=max_patients,
                max_epochs=max_epochs,
                verbose=verbose
            )
            all_results.append(result_smote)

    return all_results


def generate_comparison_table(
    results: List[Dict],
    save_path: Optional[str] = None
) -> str:
    """
    Generate a formatted comparison table from experiment results.

    Args:
        results: List of experiment result dicts.
        save_path: Optional CSV save path.

    Returns:
        Formatted table string.
    """
    lines = [
        "=" * 90,
        "  Cross-Hospital Validation Results",
        "=" * 90,
        f"{'Experiment':<35} {'AUROC':>7} {'AUPRC':>7} {'F1':>6} "
        f"{'Sens':>6} {'Spec':>6} {'SMOTE':>6}",
        "-" * 90,
    ]

    for r in results:
        m = r["metrics"]
        lines.append(
            f"{r['experiment_name']:<35} "
            f"{m.get('auroc', 0):.4f}  "
            f"{m.get('auprc', 0):.4f}  "
            f"{m.get('f1', 0):.3f}  "
            f"{m.get('sensitivity', 0):.3f}  "
            f"{m.get('specificity', 0):.3f}  "
            f"{'Yes' if r['use_smote'] else 'No':>5}"
        )

    lines.append("=" * 90)

    table_str = "\n".join(lines)
    print(table_str)

    # Save as CSV
    if save_path:
        import csv
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Experiment", "Train", "Test", "SMOTE",
                "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity",
                "Best_Val_AUROC", "Best_Epoch", "Time_s"
            ])
            for r in results:
                m = r["metrics"]
                writer.writerow([
                    r["experiment_name"],
                    r["train_hospital"],
                    r["test_hospital"],
                    r["use_smote"],
                    f"{m.get('auroc', 0):.4f}",
                    f"{m.get('auprc', 0):.4f}",
                    f"{m.get('f1', 0):.3f}",
                    f"{m.get('sensitivity', 0):.3f}",
                    f"{m.get('specificity', 0):.3f}",
                    f"{r.get('best_val_auroc', 0):.4f}",
                    r.get("best_epoch", 0),
                    f"{r.get('elapsed_seconds', 0):.1f}",
                ])
        print(f"Results saved to: {save_path}")

    return table_str
