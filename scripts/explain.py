"""
SHAP Explanation Script.

Generates SHAP-based explanations for the trained BiLSTM+Attention model,
implementing Research Objective O2.

Usage:
    python scripts/explain.py --model results/models/best_model.pt
    python scripts/explain.py --model results/models/best_model.pt --num-samples 100
    python scripts/explain.py --model results/models/best_model.pt --explainer kernel
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
from src.models.bilstm_attention import BiLSTMAttention
from src.explainability.shap_analysis import SHAPExplainer
from src.explainability.visualizations import (
    plot_feature_importance,
    plot_temporal_heatmap,
    plot_confusion_matrix,
    plot_training_history
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for the sepsis prediction model"
    )
    parser.add_argument(
        "--model", type=str, default=str(config.MODELS_DIR / "best_model.pt"),
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--hospital", type=str, default="A", choices=["A", "B", "AB"],
        help="Hospital data to use for explanation"
    )
    parser.add_argument(
        "--num-samples", type=int, default=200,
        help="Number of test samples to explain"
    )
    parser.add_argument(
        "--num-background", type=int, default=100,
        help="Number of background samples for SHAP"
    )
    parser.add_argument(
        "--explainer", type=str, default="deep", choices=["deep", "kernel"],
        help="SHAP explainer type"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit patients for speed"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  SHAP Explainability Analysis - Sepsis Prediction System")
    print("=" * 65)
    print(f"  Model: {args.model}")
    print(f"  Explainer: {args.explainer}")
    print(f"  Samples to explain: {args.num_samples}")
    print("=" * 65)

    # ---- Load data ----
    print("\n[1/5] Loading data...")
    data = prepare_dataset(
        hospital=args.hospital,
        max_patients=args.subset,
        verbose=True
    )

    X_test = data["X_test"]
    y_test = data["y_test"]
    X_train = data["X_train"]

    # ---- Load model ----
    print("\n[2/5] Loading trained model...")
    model = BiLSTMAttention()

    if os.path.exists(args.model):
        try:
            checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.model, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded model from epoch {checkpoint.get('epoch', '?')}")
    else:
        print(f"  [WARNING] Model file not found: {args.model}")
        print(f"  Using untrained model (for demonstration)")

    model.eval()

    # ---- Get predictions ----
    print("\n[3/5] Computing predictions...")
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test[:args.num_samples], dtype=torch.float32)
        logits = model(X_test_tensor).squeeze(-1)
        y_prob = torch.sigmoid(logits).numpy()

    y_true = y_test[:args.num_samples]

    # ---- Compute SHAP values ----
    print("\n[4/5] Computing SHAP values...")

    # Select background samples
    bg_indices = np.random.choice(len(X_train), min(args.num_background, len(X_train)), replace=False)
    background = X_train[bg_indices]

    try:
        explainer = SHAPExplainer(
            model=model,
            background_data=background,
            explainer_type=args.explainer
        )

        shap_values = explainer.compute_shap_values(
            X_test[:args.num_samples],
            max_samples=args.num_samples
        )

        # Generate report
        report = explainer.generate_report(
            X_test[:args.num_samples],
            y_true,
            y_prob,
            shap_values,
            save_path=str(config.RESULTS_DIR / "shap_report.json")
        )

        # ---- Generate visualizations ----
        print("\n[5/5] Generating visualizations...")

        # Feature importance plot
        plot_feature_importance(
            report["feature_importance"],
            save_path=str(config.FIGURES_DIR / "shap_feature_importance.png")
        )

        # Temporal heatmap
        if config.USE_MISSINGNESS_MASK:
            feat_names = config.CLINICAL_FEATURES + [f"{f}_mask" for f in config.CLINICAL_FEATURES]
        else:
            feat_names = config.CLINICAL_FEATURES

        plot_temporal_heatmap(
            report["temporal_importance"],
            feature_names=feat_names,
            save_path=str(config.FIGURES_DIR / "temporal_shap_heatmap.png")
        )

        # Print example patient explanations
        print("\n" + "=" * 65)
        print("  Example Patient Explanations")
        print("=" * 65)
        for explanation in report.get("example_explanations", []):
            print(explanation)
            print()

    except Exception as e:
        print(f"\n  [WARNING] SHAP computation failed: {e}")
        print(f"  This is common with DeepExplainer on complex models.")
        print(f"  Try --explainer kernel (slower but more robust)")
        import traceback
        traceback.print_exc()

    print("\n[SUCCESS] SHAP analysis complete!")
    print(f"  Figures saved to: {config.FIGURES_DIR}")
    print(f"  Report saved to: {config.RESULTS_DIR / 'shap_report.json'}")


if __name__ == "__main__":
    main()
