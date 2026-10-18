"""
SHAP Explainability Module for the Sepsis Prediction System.

Implements Research Objective O2: explain predictions in doctor-readable
terms using SHAP (SHapley Additive exPlanations).

Provides:
  - SHAP value computation via DeepExplainer or KernelExplainer
  - Global feature importance ranking
  - Patient-level explanations with clinical context
  - Temporal SHAP analysis (feature importance over time)

References:
    - Lundberg et al., "From local explanations to global understanding
      with explainable AI for trees," Nature Machine Intelligence, 2020.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


class SHAPExplainer:
    """
    SHAP-based model explainer for the BiLSTM sepsis prediction model.

    Wraps the shap library to provide both global (dataset-wide) and
    local (per-patient) explanations. Uses DeepExplainer for speed
    with fallback to KernelExplainer for robustness.
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: np.ndarray,
        feature_names: List[str] = None,
        explainer_type: str = "deep",
        device: str = "cpu"
    ):
        """
        Args:
            model: Trained PyTorch model.
            background_data: Representative background samples for SHAP,
                             shape (n_background, seq_len, n_features).
            feature_names: Names of input features.
            explainer_type: "deep" (DeepExplainer) or "kernel" (KernelExplainer).
            device: Device for computation.
        """
        import shap

        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.explainer_type = explainer_type

        # Feature names
        if feature_names is None:
            if config.USE_MISSINGNESS_MASK:
                self.feature_names = (
                    config.CLINICAL_FEATURES +
                    [f"{f}_mask" for f in config.CLINICAL_FEATURES]
                )
            else:
                self.feature_names = config.CLINICAL_FEATURES
        else:
            self.feature_names = feature_names

        # Create SHAP explainer
        if explainer_type == "deep":
            background_tensor = torch.tensor(
                background_data, dtype=torch.float32
            ).to(self.device)
            self.explainer = shap.DeepExplainer(self.model, background_tensor)
            print(f"SHAP DeepExplainer initialized with {len(background_data)} background samples")

        elif explainer_type == "kernel":
            # KernelExplainer needs a prediction function
            def predict_fn(x):
                self.model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                    if x_tensor.ndim == 2:
                        # If flattened, reshape to 3D
                        seq_len = background_data.shape[1]
                        n_features = background_data.shape[2]
                        x_tensor = x_tensor.reshape(-1, seq_len, n_features)
                    logits = self.model(x_tensor).squeeze(-1)
                    return torch.sigmoid(logits).cpu().numpy()

            # Use k-means summary for efficiency
            bg_flat = background_data.reshape(background_data.shape[0], -1)
            bg_summary = shap.kmeans(bg_flat, min(50, len(bg_flat)))
            self.explainer = shap.KernelExplainer(predict_fn, bg_summary)
            print(f"SHAP KernelExplainer initialized with k-means summary")

        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

    def compute_shap_values(
        self,
        X: np.ndarray,
        max_samples: int = None
    ) -> np.ndarray:
        """
        Compute SHAP values for input samples.

        Args:
            X: Input samples, shape (n_samples, seq_len, n_features).
            max_samples: Limit number of samples (for speed).

        Returns:
            SHAP values array, same shape as X.
        """
        if max_samples is not None and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]

        print(f"Computing SHAP values for {len(X)} samples...")

        if self.explainer_type == "deep":
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            shap_values = self.explainer.shap_values(X_tensor)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.cpu().numpy()
        else:
            X_flat = X.reshape(X.shape[0], -1)
            shap_values = self.explainer.shap_values(X_flat)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = shap_values.reshape(X.shape)

        print(f"SHAP values computed. Shape: {shap_values.shape}")
        return shap_values

    def get_feature_importance(
        self,
        shap_values: np.ndarray,
        top_k: int = 10,
        aggregate: str = "mean_abs"
    ) -> List[Tuple[str, float]]:
        """
        Rank features by their global SHAP importance.

        Aggregates SHAP values across all samples and timesteps to produce
        a single importance score per feature.

        Args:
            shap_values: SHAP values, shape (n_samples, seq_len, n_features).
            top_k: Number of top features to return.
            aggregate: Aggregation method ("mean_abs", "mean", "max_abs").

        Returns:
            List of (feature_name, importance_score) tuples, sorted descending.
        """
        n_features = shap_values.shape[-1]

        if aggregate == "mean_abs":
            # Mean of absolute SHAP values across samples and time
            importance = np.abs(shap_values).mean(axis=(0, 1))
        elif aggregate == "mean":
            importance = shap_values.mean(axis=(0, 1))
        elif aggregate == "max_abs":
            importance = np.abs(shap_values).max(axis=(0, 1))
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        # Pair with feature names
        feature_importance = []
        for i in range(min(n_features, len(self.feature_names))):
            feature_importance.append((self.feature_names[i], float(importance[i])))

        # Sort by importance descending
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        return feature_importance[:top_k]

    def get_temporal_importance(
        self,
        shap_values: np.ndarray
    ) -> np.ndarray:
        """
        Compute feature importance over time (temporal SHAP).

        For each timestep, compute the mean absolute SHAP value across
        all samples, producing a (seq_len, n_features) matrix showing
        how feature importance changes over the observation window.

        Args:
            shap_values: SHAP values, shape (n_samples, seq_len, n_features).

        Returns:
            Temporal importance matrix, shape (seq_len, n_features).
        """
        return np.abs(shap_values).mean(axis=0)

    def explain_patient(
        self,
        patient_data: np.ndarray,
        shap_values: np.ndarray,
        prediction_prob: float,
        top_k: int = 5
    ) -> str:
        """
        Generate a doctor-readable explanation for a single patient's
        sepsis risk prediction.

        This is the key clinical output for Research Objective O2:
        each alert is accompanied by a ranked list of contributing factors.

        Args:
            patient_data: Patient sequence, shape (seq_len, n_features).
            shap_values: SHAP values for this patient, same shape.
            prediction_prob: Model's predicted probability.
            top_k: Number of top contributing factors to show.

        Returns:
            Formatted explanation string.
        """
        # Get mean absolute SHAP value per feature (across time)
        feature_shap = np.abs(shap_values).mean(axis=0)

        # Get top contributing features
        top_indices = np.argsort(feature_shap)[::-1][:top_k]

        # Determine risk level
        if prediction_prob >= 0.8:
            risk_level = "HIGH RISK"
            risk_emoji = "[CRITICAL]"
        elif prediction_prob >= 0.5:
            risk_level = "MODERATE RISK"
            risk_emoji = "[WARNING]"
        else:
            risk_level = "LOW RISK"
            risk_emoji = "[INFO]"

        lines = [
            f"{'='*55}",
            f"  {risk_emoji} Sepsis Risk Assessment",
            f"{'='*55}",
            f"  Risk Score: {prediction_prob:.2f} ({risk_level})",
            f"  Prediction: {'Sepsis likely within 6 hours' if prediction_prob >= 0.5 else 'Low sepsis risk'}",
            f"",
            f"  Top Contributing Factors:",
            f"  {'-'*45}",
        ]

        for rank, idx in enumerate(top_indices, 1):
            if idx >= len(self.feature_names):
                continue

            feat_name = self.feature_names[idx]

            # Skip mask features in the clinical explanation
            if "_mask" in feat_name:
                continue

            shap_contribution = float(feature_shap[idx])

            # Get current value (last timestep)
            current_val = float(patient_data[-1, idx])

            # Get trend (compare first half vs second half)
            mid = len(patient_data) // 2
            early_val = float(np.nanmean(patient_data[:mid, idx]))
            late_val = float(np.nanmean(patient_data[mid:, idx]))

            # Determine direction
            if late_val > early_val * 1.05:
                direction = "(+)"
            elif late_val < early_val * 0.95:
                direction = "(-)"
            else:
                direction = "(=)"

            # Get clinical context
            threshold = config.CLINICAL_THRESHOLDS.get(feat_name, {})
            unit = threshold.get("unit", "")
            abnormal = ""
            if "high" in threshold and current_val > threshold["high"]:
                abnormal = " [ELEVATED]"
            elif "low" in threshold and current_val < threshold["low"]:
                abnormal = " [LOW]"

            lines.append(
                f"  {rank}. {direction} {feat_name}: {current_val:.1f} {unit}"
                f"{abnormal}  (SHAP: +{shap_contribution:.3f})"
            )

        lines.append(f"  {'-'*45}")
        lines.append(f"  Note: Higher SHAP values indicate stronger")
        lines.append(f"  contribution to the risk prediction.")
        lines.append(f"{'='*55}")

        return "\n".join(lines)

    def generate_report(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        shap_values: np.ndarray,
        save_path: str = None
    ) -> Dict:
        """
        Generate a comprehensive SHAP analysis report.

        Args:
            X: Input features.
            y_true: True labels.
            y_prob: Predicted probabilities.
            shap_values: Computed SHAP values.
            save_path: Optional path to save the report.

        Returns:
            Report dictionary with feature importance, temporal analysis, etc.
        """
        report = {}

        # Global feature importance
        report["feature_importance"] = self.get_feature_importance(
            shap_values, top_k=20
        )

        # Temporal importance
        report["temporal_importance"] = self.get_temporal_importance(shap_values)

        # Feature importance for positive vs negative cases
        pos_mask = y_true == 1
        neg_mask = y_true == 0

        if pos_mask.any():
            report["positive_importance"] = self.get_feature_importance(
                shap_values[pos_mask], top_k=10
            )
        if neg_mask.any():
            report["negative_importance"] = self.get_feature_importance(
                shap_values[neg_mask], top_k=10
            )

        # Example patient explanations (top 3 highest risk)
        top_risk_indices = np.argsort(y_prob)[::-1][:3]
        report["example_explanations"] = []
        for idx in top_risk_indices:
            explanation = self.explain_patient(
                X[idx], shap_values[idx], y_prob[idx]
            )
            report["example_explanations"].append(explanation)

        # Print summary
        print("\n" + "=" * 60)
        print("  SHAP Analysis Report")
        print("=" * 60)
        print("\nTop 10 Features by Global Importance:")
        for i, (name, score) in enumerate(report["feature_importance"][:10], 1):
            print(f"  {i:2d}. {name:25s}  {score:.4f}")

        if save_path:
            import json
            save_dict = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in report.items()
            }
            with open(save_path, "w") as f:
                json.dump(save_dict, f, indent=2, default=str)
            print(f"\nReport saved to: {save_path}")

        return report
