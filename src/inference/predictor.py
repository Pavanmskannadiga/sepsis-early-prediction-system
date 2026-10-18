"""
SepsisPredictor: Production-ready inference for single patients.

Usage:
    predictor = SepsisPredictor.from_checkpoint('results/models/best_model.pt')
    result = predictor.predict_patient('data/raw/training_setA/p000001.psv')
    print(result)  # {'risk_score': 0.87, 'risk_level': 'HIGH', 'top_features': [...]}

Design principles:
    - from_checkpoint() is the canonical factory method; it restores model
      architecture, weights, scaler, threshold, and feature names from
      a single .pt checkpoint written by SepsisTrainer.
    - All public methods accept / return plain Python / NumPy types so the
      class can be imported without touching the training pipeline.
    - NaN handling mirrors the training pre-processing: forward fill, then
      median imputation from config.POPULATION_MEDIANS.

Authors: Pavana M.S., Megharaja S., Srujan S. Sagar, Abubakkar Siddiq
Guide  : Prof. Shilpa K.C.
Dept.  : AIML, PESITM Shivamogga, 2026
"""

import sys
import math
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.models.bilstm_attention import BiLSTMAttention, build_model
from src.training.metrics import find_optimal_threshold


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

_RISK_BANDS = [
    (0.30, "LOW"),
    (0.60, "MODERATE"),
    (0.80, "HIGH"),
    (1.01, "CRITICAL"),
]

_PSV_SEPARATOR = "|"


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #

def _resolve_device(device: str) -> torch.device:
    """Resolve 'auto' to the best available device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _impute_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill then median-impute using population medians from config.

    Args:
        df: Raw DataFrame with clinical feature columns.

    Returns:
        Imputed DataFrame.
    """
    df = df.copy()
    # Forward fill within each patient (temporal carry-forward)
    df = df.ffill()
    # Fill remaining NaN with population medians
    for col, median_val in config.POPULATION_MEDIANS.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
    # Any columns not covered by POPULATION_MEDIANS: fill with 0
    df = df.fillna(0.0)
    return df


def _load_psv(psv_path: str) -> pd.DataFrame:
    """
    Read a PhysioNet PSV file into a DataFrame.

    Args:
        psv_path: Absolute or relative path to a .psv file.

    Returns:
        DataFrame where each row is one patient-hour.
    """
    path = Path(psv_path)
    if not path.exists():
        raise FileNotFoundError(f"PSV file not found: {psv_path}")
    df = pd.read_csv(path, sep=_PSV_SEPARATOR)
    return df


# --------------------------------------------------------------------------- #
# SepsisPredictor                                                             #
# --------------------------------------------------------------------------- #

class SepsisPredictor:
    """
    Production-ready inference wrapper around BiLSTMAttention.

    Handles end-to-end inference for a single patient or a batch:
        PSV file  -->  preprocessing  -->  model forward pass  -->  risk report

    Args:
        model:         Trained BiLSTMAttention instance (in eval mode).
        scaler:        Optional sklearn-compatible scaler (fit on training set).
                       If None, raw feature values are used as-is.
        threshold:     Binary classification threshold (default 0.5).
        device:        torch.device or string ('cpu', 'cuda').
        feature_names: Ordered list of feature column names expected by the
                       model.  Defaults to config.CLINICAL_FEATURES.
    """

    def __init__(
        self,
        model: nn.Module,
        scaler=None,
        threshold: float = 0.5,
        device: Union[str, torch.device] = "cpu",
        feature_names: Optional[List[str]] = None,
        model_type: str = "bilstm_attention"
    ):
        self.device   = _resolve_device(device) if isinstance(device, str) else device
        self.model    = model.to(self.device)
        self.model.eval()

        self.scaler        = scaler
        self.threshold     = threshold
        self.feature_names = feature_names or config.CLINICAL_FEATURES
        self.seq_len       = config.SEQUENCE_LENGTH
        self.model_type    = model_type

    # ------------------------------------------------------------------ #
    # Factory method                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto"
    ) -> "SepsisPredictor":
        """
        Load a SepsisPredictor from a .pt checkpoint file.

        The checkpoint is expected to follow the format written by
        SepsisTrainer.save_checkpoint(), which stores:
            - 'model_state_dict'  : model weights
            - 'model_config'      : dict of BiLSTMAttention constructor kwargs
            - 'threshold'         : optimal classification threshold
            - 'scaler'            : sklearn scaler object (or None)
            - 'feature_names'     : list of feature column names (or None)
            - 'epoch'             : training epoch at which checkpoint was saved

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            device:          Target device ('auto', 'cpu', 'cuda').

        Returns:
            Fully initialised SepsisPredictor ready for inference.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        resolved = _resolve_device(device)

        # Load checkpoint -- support both PyTorch >=2.0 and older versions
        try:
            ckpt = torch.load(
                checkpoint_path, map_location=resolved, weights_only=False
            )
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=resolved)

        # Determine model type from checkpoint
        model_type = ckpt.get("model_type", "bilstm_attention")

        # Rebuild model architecture
        model_config = ckpt.get("model_config", {})
        model = build_model(model_type, **model_config)
        model.load_state_dict(ckpt["model_state_dict"])

        threshold     = ckpt.get("threshold",     0.5)
        scaler        = ckpt.get("scaler",        None)
        feature_names = ckpt.get("feature_names", None)

        epoch = ckpt.get("epoch", "?")
        model_names = {
            "bilstm_attention": "BiLSTM+Attention",
            "transformer": "Transformer Encoder",
            "tcn": "TCN",
        }
        print(
            f"[SepsisPredictor] Loaded {model_names.get(model_type, model_type)} "
            f"checkpoint from epoch {epoch} "
            f"| threshold={threshold:.4f} | device={resolved}"
        )

        return cls(
            model=model,
            scaler=scaler,
            threshold=threshold,
            device=resolved,
            feature_names=feature_names,
            model_type=model_type,
        )

    # ------------------------------------------------------------------ #
    # Core inference                                                      #
    # ------------------------------------------------------------------ #

    def predict_from_array(
        self,
        X: np.ndarray
    ) -> Dict:
        """
        Run inference on a pre-processed numpy array.

        Args:
            X: Shape (seq_len, n_features) for a single patient, or
               (batch, seq_len, n_features) for a batch.
               Values should already be normalised if a scaler was used.

        Returns:
            Dict with:
                'risk_score'  : float, mean probability across timesteps
                'risk_level'  : str, one of LOW / MODERATE / HIGH / CRITICAL
                'prediction'  : int, binary label (0 or 1)
                'probability' : float, alias for risk_score
        """
        if X.ndim == 2:
            # Single patient: (seq_len, features) -> (1, seq_len, features)
            X = X[np.newaxis, :, :]

        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor, return_attention=True)   # (B, 1)
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()  # (B,)

        mean_prob = float(probs.mean())
        level     = self.risk_level(mean_prob)
        pred      = int(mean_prob >= self.threshold)

        return {
            "risk_score" : mean_prob,
            "probability": mean_prob,
            "risk_level" : level,
            "prediction" : pred,
        }

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Batch inference; returns raw probability array.

        Args:
            X: Shape (batch, seq_len, n_features).

        Returns:
            1-D numpy array of probabilities, shape (batch,).
        """
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)  # (B, 1)
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()

        return probs

    # ------------------------------------------------------------------ #
    # Patient-level inference from PSV                                   #
    # ------------------------------------------------------------------ #

    def predict_patient(self, psv_file_path: str) -> Dict:
        """
        Full pipeline: read PSV, preprocess, run model, return risk dict.

        Args:
            psv_file_path: Path to a PhysioNet-format .psv file.  Each row
                           represents one observation hour for the patient.

        Returns:
            Dict with:
                'patient_id'    : str (derived from filename)
                'risk_score'    : float  [0, 1]
                'risk_level'    : str    LOW / MODERATE / HIGH / CRITICAL
                'prediction'    : int    0 or 1
                'iculos_hours'  : int    number of ICU hours in file
                'sepsis_label'  : int or None (from file if present)
                'top_features'  : list of (feature_name, attention_weight) tuples
                'feature_means' : dict of feature -> mean value over window
        """
        path = Path(psv_file_path)
        df   = _load_psv(str(path))

        # Extract ground truth label if available
        sepsis_label: Optional[int] = None
        if config.TARGET_COLUMN in df.columns:
            sepsis_label = int(df[config.TARGET_COLUMN].max())
            df = df.drop(columns=[config.TARGET_COLUMN])

        iculos_hours = len(df)

        # Pre-process
        X, feature_means = self._preprocess_patient(df)

        # Forward pass
        tensor = torch.tensor(
            X[np.newaxis, :, :], dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor, return_attention=True)   # (1, 1)
            prob   = float(torch.sigmoid(logits).item())

        # Extract attention weights for top-feature ranking
        top_features = self._get_top_features(tensor)

        result = {
            "patient_id"   : path.stem,
            "risk_score"   : prob,
            "probability"  : prob,
            "risk_level"   : self.risk_level(prob),
            "prediction"   : int(prob >= self.threshold),
            "iculos_hours" : iculos_hours,
            "sepsis_label" : sepsis_label,
            "top_features" : top_features,
            "feature_means": feature_means,
        }

        return result

    # ------------------------------------------------------------------ #
    # Preprocessing                                                       #
    # ------------------------------------------------------------------ #

    def _preprocess_patient(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Align columns, impute NaN, optionally scale, pad/trim to seq_len.

        Returns:
            Tuple of (X_array, feature_means_dict)
            X_array shape: (seq_len, n_features)   -- model input
        """
        # Keep only the clinical feature columns in the correct order
        clinical_cols = [c for c in self.feature_names if c in df.columns]
        missing_cols  = [c for c in self.feature_names if c not in df.columns]

        df_clinical = df[clinical_cols].copy()

        # Add any completely missing columns as NaN then impute
        for col in missing_cols:
            df_clinical[col] = np.nan

        # Reorder to match feature_names exactly
        df_clinical = df_clinical[self.feature_names]

        # Impute
        df_imputed = _impute_series(df_clinical)

        # Feature means (for report)
        feature_means = {col: float(df_imputed[col].mean())
                         for col in self.feature_names}

        # Append missingness mask (column-wise binary indicator)
        if config.USE_MISSINGNESS_MASK:
            mask_df = df_clinical.isna().astype(float)
            mask_df.columns = [c + "_mask" for c in self.feature_names]
            df_imputed = pd.concat([df_imputed, mask_df], axis=1)

        X = df_imputed.values.astype(np.float32)   # (n_hours, n_features)

        # Optional scaling
        if self.scaler is not None:
            orig_shape = X.shape
            X_2d = X.reshape(-1, X.shape[-1])
            X_2d = self.scaler.transform(X_2d)
            X    = X_2d.reshape(orig_shape)

        # Pad or trim to seq_len
        n_hours = X.shape[0]
        if n_hours >= self.seq_len:
            # Use the most recent seq_len hours
            X = X[-self.seq_len:, :]
        else:
            # Left-pad with zeros
            pad = np.zeros((self.seq_len - n_hours, X.shape[1]), dtype=np.float32)
            X   = np.concatenate([pad, X], axis=0)

        return X, feature_means

    # ------------------------------------------------------------------ #
    # Attention-based feature importance                                  #
    # ------------------------------------------------------------------ #

    def _get_top_features(
        self,
        tensor: torch.Tensor,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Derive top-k important features from attention weights (if available)
        or from gradient-based importance as a fallback for models without
        attention (e.g., TCN).

        Args:
            tensor: Input tensor already on device, shape (1, seq_len, n_feat).
            top_k:  Number of top features to return.

        Returns:
            List of (feature_name, weight) pairs sorted descending.
            Returns empty list if importance cannot be computed.
        """
        attn_weights = self.model.get_attention_weights()

        if attn_weights is not None:
            # Attention-based feature importance (BiLSTM / Transformer)
            # attn_weights: (1, heads, seq_len, seq_len) or similar
            attn = attn_weights[0].mean(dim=0).cpu().numpy()  # (seq, seq)

            # Mean over query axis -> per-timestep weight (seq,)
            timestep_importance = attn.mean(axis=0)
            timestep_importance = timestep_importance / (timestep_importance.sum() + 1e-9)

            # For Transformer, attn includes CLS token — skip it
            X_np = tensor[0].cpu().numpy()  # (seq_len, n_features)
            if len(timestep_importance) > X_np.shape[0]:
                # Trim CLS token position (first position)
                timestep_importance = timestep_importance[1:]

            n_feat = min(len(self.feature_names), X_np.shape[1])
            weighted = (timestep_importance[:, np.newaxis] * X_np[:, :n_feat]).sum(axis=0)
        else:
            # Gradient-based fallback (for TCN and other models without attention)
            # Use absolute value of last-timestep features as a simple proxy
            X_np = tensor[0].cpu().numpy()  # (seq_len, n_features)
            n_feat = min(len(self.feature_names), X_np.shape[1])
            # Use variance over time × absolute last value as importance
            feat_var = np.var(X_np[:, :n_feat], axis=0)
            feat_last = np.abs(X_np[-1, :n_feat])
            weighted = feat_var * feat_last

        feat_names = self.feature_names[:n_feat]
        ranked = sorted(
            zip(feat_names, weighted.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------ #
    # Risk stratification                                                 #
    # ------------------------------------------------------------------ #

    def risk_level(self, probability: float) -> str:
        """
        Categorise a probability into a clinical risk band.

        Bands:
            LOW      : p < 0.30   (routine monitoring)
            MODERATE : 0.30 <= p < 0.60  (increased vigilance)
            HIGH     : 0.60 <= p < 0.80  (urgent review)
            CRITICAL : p >= 0.80  (immediate intervention)

        Args:
            probability: Predicted sepsis probability in [0, 1].

        Returns:
            Risk level string.
        """
        for upper, label in _RISK_BANDS:
            if probability < upper:
                return label
        return "CRITICAL"

    # ------------------------------------------------------------------ #
    # Clinical report generation                                         #
    # ------------------------------------------------------------------ #

    def generate_report(self, psv_file_path: str) -> str:
        """
        Generate a human-readable clinical risk report for a patient.

        Args:
            psv_file_path: Path to a PhysioNet .psv patient file.

        Returns:
            Multi-line formatted report string suitable for console output
            or plain-text logging.
        """
        result = self.predict_patient(psv_file_path)

        prob    = result["risk_score"]
        level   = result["risk_level"]
        pred    = result["prediction"]
        patient = result["patient_id"]
        hours   = result["iculos_hours"]
        true_lbl = result["sepsis_label"]
        fm       = result["feature_means"]
        tf       = result["top_features"]

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Risk level banner
        banners = {
            "LOW"     : "[ LOW RISK      ]",
            "MODERATE": "[ MODERATE RISK ]",
            "HIGH"    : "[ HIGH RISK     ]",
            "CRITICAL": "[ CRITICAL RISK ]",
        }
        banner = banners.get(level, "[ UNKNOWN       ]")

        lines = [
            "=" * 65,
            f"  SEPSIS PREDICTION REPORT",
            f"  Generated: {now}",
            "=" * 65,
            f"  Patient ID      : {patient}",
            f"  ICU Hours       : {hours}",
            f"  Risk Score      : {prob:.4f}  ({prob * 100:.1f}%)",
            f"  Risk Level      : {banner}",
            f"  Prediction      : {'SEPSIS PREDICTED' if pred else 'No sepsis predicted'}",
            f"  Threshold       : {self.threshold:.4f}",
        ]

        if true_lbl is not None:
            correct = (pred == true_lbl)
            lines.append(
                f"  True Label      : {true_lbl}  "
                f"({'Correct' if correct else 'Incorrect'})"
            )

        lines += [
            "-" * 65,
            "  KEY VITAL SIGNS (mean over observation window):",
        ]

        vitals_to_show = [
            ("HR",    "Heart Rate",   "bpm"),
            ("Resp",  "Resp Rate",    "br/min"),
            ("O2Sat", "O2 Sat",       "%"),
            ("SBP",   "Systolic BP",  "mmHg"),
            ("MAP",   "MAP",          "mmHg"),
            ("Temp",  "Temperature",  "degC"),
            ("Lactate","Lactate",     "mmol/L"),
        ]
        for col, label, unit in vitals_to_show:
            if col in fm:
                ref = config.CLINICAL_THRESHOLDS.get(col, {})
                val = fm[col]
                flag = ""
                if ref:
                    if val < ref.get("low", -math.inf):
                        flag = " [LOW]"
                    elif val > ref.get("high", math.inf):
                        flag = " [HIGH]"
                lines.append(f"    {label:15s}: {val:8.2f} {unit}{flag}")

        if tf:
            lines += [
                "-" * 65,
                f"  TOP PREDICTIVE FEATURES (attention-weighted):",
            ]
            for rank, (feat, weight) in enumerate(tf[:10], start=1):
                lines.append(f"    {rank:2d}. {feat:20s}  weight={weight:+.4f}")

        lines += [
            "=" * 65,
            "  DISCLAIMER: AI decision support only. Clinical judgement",
            "  must always supersede algorithmic predictions.",
            "=" * 65,
            "",
        ]

        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Quick self-test                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("SepsisPredictor self-test (no checkpoint, synthetic data)")
    print("-" * 55)

    # Build a model with default config
    model = BiLSTMAttention()
    predictor = SepsisPredictor(model=model, threshold=0.5, device="cpu", model_type="bilstm_attention")

    # Synthetic patient input: 48 hours x n_features
    n_feat = config.NUM_INPUT_FEATURES
    X_fake = np.random.randn(config.SEQUENCE_LENGTH, n_feat).astype(np.float32)

    result = predictor.predict_from_array(X_fake)
    print(f"Risk score : {result['risk_score']:.4f}")
    print(f"Risk level : {result['risk_level']}")
    print(f"Prediction : {result['prediction']}")
    print("Self-test passed.")
