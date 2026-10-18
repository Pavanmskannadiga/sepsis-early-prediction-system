"""
Clinical Baseline Scores for Sepsis Prediction Comparison.

Implements three widely-used clinical risk scoring systems that serve as
interpretable baselines to benchmark the BiLSTM+Attention model against:

  - qSOFA  (quick Sequential Organ Failure Assessment, 0–3)
  - SOFA   (Sequential Organ Failure Assessment, partial, 0–20)
  - NEWS2  (National Early Warning Score 2, 0–20)

These scores are computed row-by-row from PhysioNet/CinC 2019 features,
mirroring the signals available to a bedside clinician, and evaluated with
the same compute_all_metrics() function used by the ML pipeline so that
all models are compared on a level playing field.

Research context:
    Addresses Research Objective O1 comparison: demonstrating that our
    BiLSTM model improves on rule-based clinical scores, which were
    identified as a key limitation in prior work (Survey gap G1).

Available PhysioNet/CinC 2019 feature columns used here:
    Vitals      : HR, O2Sat, Temp, SBP, MAP, DBP, Resp
    Labs        : Creatinine, Bilirubin_total, WBC, Platelets,
                  pH, PaCO2, FiO2, Lactate, Hct, HCO3
    Demographics: Age, ICULOS

Missing value policy:
    If a required field is NaN for a given row, that sub-component
    contributes 0 points (safe / conservative default).

Authors: Pavana M.S., Megharaja S., Srujan S. Sagar, Abubakkar Siddiq
Guide  : Prof. Shilpa K.C.
Dept.  : AIML, PESITM Shivamogga, 2026
"""

import sys
import csv
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.training.metrics import compute_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _safe(value) -> Optional[float]:
    """Return float or None if value is NaN / missing."""
    try:
        v = float(value)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. qSOFA
# ─────────────────────────────────────────────────────────────────────────────

def compute_qsofa_score(row: pd.Series) -> int:
    """
    Compute the quick SOFA (qSOFA) score for a single patient row.

    The original qSOFA criteria (Singer et al., 2016):
        1. Respiratory rate >= 22 breaths/min
        2. Systolic BP <= 100 mmHg
        3. Altered mentation (GCS < 15)

    Because the PhysioNet dataset does not contain a GCS column, Age >= 70
    is used as a conservative proxy for altered mentation risk, consistent
    with published ICU risk-score adaptations for claims/registry data.

    Args:
        row: A pandas Series representing one patient-hour observation.
             Expected columns: Resp, SBP, Age.

    Returns:
        Integer score in [0, 3].  Missing components score 0.
    """
    score = 0

    resp = _safe(row.get("Resp"))
    sbp  = _safe(row.get("SBP"))
    age  = _safe(row.get("Age"))

    # Criterion 1: High respiratory rate
    if resp is not None and resp >= 22:
        score += 1

    # Criterion 2: Low systolic blood pressure
    if sbp is not None and sbp <= 100:
        score += 1

    # Criterion 3: Altered mentation proxy (age >= 70)
    if age is not None and age >= 70:
        score += 1

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 2. SOFA (partial)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sofa_score(row: pd.Series) -> int:
    """
    Compute a partial SOFA score for a single patient-hour observation.

    Scored subsystems (omitting GCS / vasopressor dose which are absent):
        Respiratory    - SpO2/FiO2 ratio as proxy for PaO2/FiO2
        Coagulation    - Platelet count (x10^3/uL)
        Liver          - Total bilirubin (mg/dL)
        Renal          - Creatinine (mg/dL)
        Cardiovascular - MAP (mmHg)

    Reference thresholds follow the original SOFA paper
    (Vincent et al., 1996) and Seymour et al., 2016.

    Args:
        row: A pandas Series representing one patient-hour observation.
             Expected columns: O2Sat, FiO2, Platelets, Bilirubin_total,
             Creatinine, MAP.

    Returns:
        Integer score in [0, ~13 with partial scoring].
    """
    score = 0

    # -- Respiratory: SpO2 / FiO2 (proxy for PaO2/FiO2) ----------------------
    o2sat = _safe(row.get("O2Sat"))
    fio2  = _safe(row.get("FiO2"))

    if o2sat is not None and fio2 is not None and fio2 > 0:
        ratio = o2sat / fio2      # SpO2/FiO2 approximates PaO2/FiO2
        if ratio < 200:
            score += 3
        elif ratio < 300:
            score += 2
        elif ratio < 400:
            score += 1
        # else: 0

    # -- Coagulation: Platelets -----------------------------------------------
    plt = _safe(row.get("Platelets"))
    if plt is not None:
        if plt < 50:
            score += 3
        elif plt < 100:
            score += 2
        elif plt < 150:
            score += 1

    # -- Liver: Total Bilirubin -----------------------------------------------
    bili = _safe(row.get("Bilirubin_total"))
    if bili is not None:
        if bili > 12:
            score += 3
        elif bili > 6:
            score += 2
        elif bili > 2:
            score += 1

    # -- Renal: Creatinine ----------------------------------------------------
    creat = _safe(row.get("Creatinine"))
    if creat is not None:
        if creat > 5.0:
            score += 3
        elif creat > 3.4:
            score += 2
        elif creat > 2.0:
            score += 1

    # -- Cardiovascular: MAP --------------------------------------------------
    map_ = _safe(row.get("MAP"))
    if map_ is not None and map_ < 65:
        score += 1

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 3. NEWS2
# ─────────────────────────────────────────────────────────────────────────────

def compute_news2_score(row: pd.Series) -> int:
    """
    Compute the National Early Warning Score 2 (NEWS2) for one patient-hour.

    Scored parameters and breakpoints follow the Royal College of Physicians
    NEWS2 guidance (2017):
        Respiration rate  (breaths/min)
        Oxygen saturation (%)            -- Scale 1 used
        Systolic blood pressure (mmHg)
        Heart rate        (bpm)
        Temperature       (degC)

    Note: Consciousness (AVPU/GCS) and supplemental O2 sub-scores are
    omitted because the PhysioNet feature set does not include them.

    Args:
        row: A pandas Series representing one patient-hour observation.
             Expected columns: Resp, O2Sat, SBP, HR, Temp.

    Returns:
        Integer score in [0, 20].  Missing components score 0.
    """
    score = 0

    # -- Respiration rate -----------------------------------------------------
    resp = _safe(row.get("Resp"))
    if resp is not None:
        if resp <= 8:
            score += 3
        elif resp <= 11:        # 9-11
            score += 1
        elif resp <= 20:        # 12-20
            score += 0
        elif resp <= 24:        # 21-24
            score += 2
        else:                   # >= 25
            score += 3

    # -- Oxygen saturation (Scale 1) ------------------------------------------
    o2sat = _safe(row.get("O2Sat"))
    if o2sat is not None:
        if o2sat <= 91:
            score += 3
        elif o2sat <= 93:       # 92-93
            score += 2
        elif o2sat <= 95:       # 94-95
            score += 1
        # else >= 96: 0

    # -- Systolic blood pressure ----------------------------------------------
    sbp = _safe(row.get("SBP"))
    if sbp is not None:
        if sbp <= 90:
            score += 3
        elif sbp <= 100:        # 91-100
            score += 2
        elif sbp <= 110:        # 101-110
            score += 1
        elif sbp <= 219:        # 111-219
            score += 0
        else:                   # >= 220
            score += 3

    # -- Heart rate -----------------------------------------------------------
    hr = _safe(row.get("HR"))
    if hr is not None:
        if hr <= 40:
            score += 3
        elif hr <= 50:          # 41-50
            score += 1
        elif hr <= 90:          # 51-90
            score += 0
        elif hr <= 110:         # 91-110
            score += 1
        elif hr <= 130:         # 111-130
            score += 2
        else:                   # >= 131
            score += 3

    # -- Temperature ----------------------------------------------------------
    temp = _safe(row.get("Temp"))
    if temp is not None:
        if temp <= 35.0:
            score += 3
        elif temp <= 36.0:      # 35.1-36
            score += 1
        elif temp <= 38.0:      # 36.1-38
            score += 0
        elif temp <= 39.0:      # 38.1-39
            score += 1
        else:                   # >= 39.1
            score += 2

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluate all baselines against ground-truth labels
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clinical_baselines(
    df_raw: pd.DataFrame,
    y_true: np.ndarray
) -> Dict[str, Dict]:
    """
    Apply all three clinical scores row-by-row and evaluate against labels.

    Binary classification thresholds (clinically accepted):
        qSOFA  >= 2  ->  sepsis-positive prediction
        SOFA   >= 2  ->  sepsis-positive prediction
        NEWS2  >= 5  ->  sepsis-positive prediction

    Args:
        df_raw: Raw feature DataFrame.  Each row is one patient-hour.
                Must contain the PhysioNet/CinC 2019 feature columns.
        y_true: Ground-truth binary labels, shape (n_rows,).

    Returns:
        Dict with keys 'qSOFA', 'SOFA', 'NEWS2'.  Each value is a dict:
            'metrics'  : output of compute_all_metrics()
            'scores'   : raw integer score array (np.ndarray)
            'binary'   : binary prediction array  (np.ndarray)
            'threshold': the score threshold used for binarisation
    """
    if len(df_raw) != len(y_true):
        raise ValueError(
            f"df_raw has {len(df_raw)} rows but y_true has {len(y_true)} elements."
        )

    # -- Compute raw scores ---------------------------------------------------
    print("[Baselines] Computing qSOFA scores ...", flush=True)
    qsofa_scores = np.array(
        [compute_qsofa_score(row) for _, row in df_raw.iterrows()],
        dtype=np.float32
    )

    print("[Baselines] Computing SOFA scores ...",  flush=True)
    sofa_scores = np.array(
        [compute_sofa_score(row) for _, row in df_raw.iterrows()],
        dtype=np.float32
    )

    print("[Baselines] Computing NEWS2 scores ...", flush=True)
    news2_scores = np.array(
        [compute_news2_score(row) for _, row in df_raw.iterrows()],
        dtype=np.float32
    )

    # -- Define thresholds and scoring configs --------------------------------
    scoring_configs = {
        "qSOFA": {
            "scores"   : qsofa_scores,
            "threshold": 2,
            "max_score": 3,
        },
        "SOFA": {
            "scores"   : sofa_scores,
            "threshold": 2,
            "max_score": 20,   # partial scoring, practical max ~13
        },
        "NEWS2": {
            "scores"   : news2_scores,
            "threshold": 5,
            "max_score": 20,
        },
    }

    results: Dict[str, Dict] = {}

    for name, cfg in scoring_configs.items():
        scores    = cfg["scores"]
        threshold = cfg["threshold"]

        # Binary predictions
        binary = (scores >= threshold).astype(int)

        # Normalise scores to [0, 1] to serve as a pseudo-probability
        # for AUROC computation (rank-preserving monotone transform).
        max_score   = cfg["max_score"]
        pseudo_prob = np.clip(scores / max_score, 0.0, 1.0).astype(np.float64)

        # Compute metrics
        metrics = compute_all_metrics(
            y_true=y_true.astype(int),
            y_prob=pseudo_prob,
            threshold=threshold / max_score   # equivalent probability threshold
        )

        results[name] = {
            "metrics"  : metrics,
            "scores"   : scores,
            "binary"   : binary,
            "threshold": threshold,
        }

        print(
            f"[Baselines] {name:6s} -- AUROC={metrics['auroc']:.3f}  "
            f"F1={metrics['f1']:.3f}  "
            f"Sens={metrics['sensitivity']:.3f}  "
            f"Spec={metrics['specificity']:.3f}"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_ml_model(
    clinical_results: Dict,
    ml_metrics: Dict,
    save_path: Optional[str] = None
) -> str:
    """
    Produce a formatted ASCII comparison table: clinical baselines vs ML model.

    Columns: Model | AUROC | AUPRC | F1 | Sensitivity | Specificity | MCC

    Args:
        clinical_results: Output of evaluate_clinical_baselines().
        ml_metrics:       metrics dict from compute_all_metrics() for the ML
                          model (keys: auroc, auprc, f1, sensitivity,
                          specificity, mcc).
        save_path:        If given, also write a CSV to this path.

    Returns:
        Formatted ASCII table string (printable / loggable).
    """
    # -- Assemble rows --------------------------------------------------------
    headers = ["Model", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "MCC"]

    rows = []
    for name, result in clinical_results.items():
        m = result["metrics"]
        rows.append({
            "Model"      : name,
            "AUROC"      : m.get("auroc",       0.0),
            "AUPRC"      : m.get("auprc",       0.0),
            "F1"         : m.get("f1",          0.0),
            "Sensitivity": m.get("sensitivity", 0.0),
            "Specificity": m.get("specificity", 0.0),
            "MCC"        : m.get("mcc",         0.0),
        })

    # ML model as final row
    rows.append({
        "Model"      : "BiLSTM+Attention",
        "AUROC"      : ml_metrics.get("auroc",       0.0),
        "AUPRC"      : ml_metrics.get("auprc",       0.0),
        "F1"         : ml_metrics.get("f1",          0.0),
        "Sensitivity": ml_metrics.get("sensitivity", 0.0),
        "Specificity": ml_metrics.get("specificity", 0.0),
        "MCC"        : ml_metrics.get("mcc",         0.0),
    })

    # -- Format ASCII table ---------------------------------------------------
    col_widths = {
        "Model"      : 18,
        "AUROC"      : 8,
        "AUPRC"      : 8,
        "F1"         : 8,
        "Sensitivity": 12,
        "Specificity": 12,
        "MCC"        : 8,
    }

    def _fmt_val(h: str, v) -> str:
        if h == "Model":
            return str(v).ljust(col_widths[h])
        return f"{float(v):.4f}".ljust(col_widths[h])

    separator = "+" + "+".join("-" * (col_widths[h] + 2) for h in headers) + "+"

    header_line = (
        "| "
        + " | ".join(h.ljust(col_widths[h]) for h in headers)
        + " |"
    )

    lines = [
        "",
        "=" * len(separator),
        "  Clinical Baselines vs BiLSTM+Attention Model",
        "  (PhysioNet/CinC 2019 Sepsis Dataset)",
        "=" * len(separator),
        separator,
        header_line,
        separator,
    ]

    for i, row in enumerate(rows):
        # Insert a divider before the ML model row
        if i == len(rows) - 1:
            lines.append(separator)

        row_line = (
            "| "
            + " | ".join(_fmt_val(h, row[h]) for h in headers)
            + " |"
        )
        lines.append(row_line)

    lines.append(separator)
    lines.append("")

    table_str = "\n".join(lines)

    # -- Optional CSV export --------------------------------------------------
    if save_path is not None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        with open(save_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[Baselines] Comparison table saved to: {save_file}")

    return table_str


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke-test with a synthetic 10-row DataFrame
    import random
    random.seed(42)
    np.random.seed(42)

    n = 10
    synthetic = pd.DataFrame({
        "HR"             : np.random.uniform(50, 140, n),
        "O2Sat"          : np.random.uniform(88,  99, n),
        "Temp"           : np.random.uniform(35,  40, n),
        "SBP"            : np.random.uniform(80, 160, n),
        "MAP"            : np.random.uniform(55, 100, n),
        "DBP"            : np.random.uniform(45,  90, n),
        "Resp"           : np.random.uniform(10,  30, n),
        "FiO2"           : np.random.uniform(0.21, 1.0, n),
        "Platelets"      : np.random.uniform(40, 400, n),
        "Bilirubin_total": np.random.uniform(0.1, 15, n),
        "Creatinine"     : np.random.uniform(0.5,  6, n),
        "Age"            : np.random.uniform(30,  90, n),
        "ICULOS"         : np.random.uniform(1,   72, n),
    })
    synthetic.loc[2, "O2Sat"] = float("nan")   # inject a NaN

    y_fake = np.random.randint(0, 2, n)

    results = evaluate_clinical_baselines(synthetic, y_fake)

    # Dummy ML metrics
    ml_m = {
        "auroc": 0.92, "auprc": 0.61, "f1": 0.58,
        "sensitivity": 0.72, "specificity": 0.94, "mcc": 0.55
    }

    table = compare_with_ml_model(results, ml_m)
    print(table)
