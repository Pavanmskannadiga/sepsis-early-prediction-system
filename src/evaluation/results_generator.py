"""
Results Generator Module.

Generates all tables and figures for the research paper/report,
combining metrics from training, SHAP analysis, and cross-hospital
validation into publication-ready outputs.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def generate_model_comparison_table(
    bilstm_metrics: Dict,
    baseline_results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a comparison table of BiLSTM+Attention vs baseline models.

    Args:
        bilstm_metrics: Metrics from the BiLSTM model.
        baseline_results: Dict of baseline model name → metrics.
        save_path: Optional CSV save path.

    Returns:
        Pandas DataFrame with comparison.
    """
    rows = []

    # Add BiLSTM result
    rows.append({
        "Model": "BiLSTM+Attention (Ours)",
        "AUROC": bilstm_metrics.get("auroc", 0),
        "AUPRC": bilstm_metrics.get("auprc", 0),
        "F1": bilstm_metrics.get("f1", 0),
        "Sensitivity": bilstm_metrics.get("sensitivity", 0),
        "Specificity": bilstm_metrics.get("specificity", 0),
        "MCC": bilstm_metrics.get("mcc", 0),
    })

    # Add baseline results
    for name, metrics in baseline_results.items():
        if "error" not in metrics:
            rows.append({
                "Model": name,
                "AUROC": metrics.get("auroc", 0),
                "AUPRC": metrics.get("auprc", 0),
                "F1": metrics.get("f1", 0),
                "Sensitivity": metrics.get("sensitivity", 0),
                "Specificity": metrics.get("specificity", 0),
                "MCC": metrics.get("mcc", 0),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("AUROC", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 85)
    print("  Model Performance Comparison")
    print("=" * 85)
    print(df.to_string(index=False, float_format="%.4f"))
    print("=" * 85)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved to: {save_path}")

    return df


def generate_shap_feature_table(
    feature_importance: List[tuple],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a table of SHAP feature importance scores.

    Args:
        feature_importance: List of (feature_name, importance) tuples.
        save_path: Optional save path.

    Returns:
        DataFrame with feature rankings.
    """
    rows = []
    for rank, (name, score) in enumerate(feature_importance, 1):
        clinical_info = config.CLINICAL_THRESHOLDS.get(name, {})
        rows.append({
            "Rank": rank,
            "Feature": name,
            "Mean |SHAP|": score,
            "Normal Range": f"{clinical_info.get('low', '?')}-{clinical_info.get('high', '?')}" if clinical_info else "—",
            "Unit": clinical_info.get("unit", "—"),
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("  SHAP Feature Importance Ranking")
    print("=" * 70)
    print(df.to_string(index=False, float_format="%.4f"))
    print("=" * 70)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved to: {save_path}")

    return df


def generate_full_report(
    results: Dict,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Generate all result tables and save them.

    Args:
        results: Dictionary containing all experiment results.
        save_dir: Directory to save outputs.

    Returns:
        Dictionary of generated DataFrames.
    """
    if save_dir is None:
        save_dir = str(config.TABLES_DIR)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    report = {}

    # Model comparison
    if "bilstm_metrics" in results and "baseline_results" in results:
        report["model_comparison"] = generate_model_comparison_table(
            results["bilstm_metrics"],
            results["baseline_results"],
            save_path=str(save_dir / "model_comparison.csv")
        )

    # SHAP features
    if "feature_importance" in results:
        report["shap_features"] = generate_shap_feature_table(
            results["feature_importance"],
            save_path=str(save_dir / "shap_feature_importance.csv")
        )

    # Save full results as JSON
    results_json = str(save_dir / "full_results.json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, pd.DataFrame):
            serializable[k] = v.to_dict()
        else:
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)

    with open(results_json, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nFull results saved to: {results_json}")

    return report
