"""
Exploratory Data Analysis (EDA) for the PhysioNet/CinC 2019 Sepsis Dataset.

Generates a comprehensive suite of visualisations and statistical summaries
to understand the dataset before model training.  All plots are saved to disk;
no interactive display is required (suitable for headless servers).

Usage examples
--------------
# Analyse Hospital A, up to 500 patients, default output directory:
    python scripts/eda.py --hospital A --max-patients 500

# Analyse both hospitals, all patients, custom output:
    python scripts/eda.py --hospital both --output-dir results/figures/eda

# Analyse Hospital B only:
    python scripts/eda.py --hospital B --max-patients 1000

Authors   : Pavana M.S., Megharaja S., Srujan S. Sagar, Abubakkar Siddiq
Guide     : Prof. Shilpa K.C.
Dept.     : AIML, PESITM Shivamogga, 2026
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Project-root import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Plot aesthetics
# ---------------------------------------------------------------------------
plt.style.use("dark_background")

PALETTE_NORMAL = "#4FC3F7"   # Light blue  – non-sepsis
PALETTE_SEPSIS = "#EF5350"   # Red         – sepsis
PALETTE_BOTH   = [PALETTE_NORMAL, PALETTE_SEPSIS]

FIGURE_DPI   = 150
FIGURE_EXT   = "png"

# Key lab values to spotlight in boxplots
KEY_LABS: List[str] = ["Lactate", "Creatinine", "WBC", "BUN", "pH"]


# ===========================================================================
# Data Loading
# ===========================================================================

def _load_psv_directory(
    directory: Path,
    max_patients: int,
    hospital_label: str,
) -> Optional[pd.DataFrame]:
    """
    Load up to *max_patients* PSV files from *directory* into a single
    ``pd.DataFrame``.  Each file represents one ICU patient stay.  A
    ``patient_id`` column (filename stem) and a ``hospital`` column are added.

    Parameters
    ----------
    directory    : Path to the folder containing ``*.psv`` files.
    max_patients : Maximum number of patient files to load.
    hospital_label : Short label to tag rows, e.g. ``"A"`` or ``"B"``.

    Returns
    -------
    DataFrame or ``None`` if the directory is missing / empty.
    """
    if not directory.exists():
        return None

    psv_files = sorted(directory.glob("*.psv"))
    if not psv_files:
        return None

    psv_files = psv_files[:max_patients]
    frames: List[pd.DataFrame] = []

    for psv_path in psv_files:
        try:
            df = pd.read_csv(psv_path, sep="|")
            df["patient_id"] = psv_path.stem
            df["hospital"]   = hospital_label
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] Could not read {psv_path.name}: {exc}")

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def load_data(
    hospital: str,
    max_patients: int,
) -> Tuple[Optional[pd.DataFrame], Dict[str, int]]:
    """
    Load patient data for the requested hospital(s).

    Parameters
    ----------
    hospital     : ``"A"``, ``"B"``, or ``"both"``.
    max_patients : Maximum patients per hospital.

    Returns
    -------
    Tuple of (combined DataFrame or None, meta dict with file counts).
    """
    meta: Dict[str, int] = {"A": 0, "B": 0}
    frames: List[pd.DataFrame] = []

    hospitals_to_load: Dict[str, Path] = {}
    if hospital in ("A", "both"):
        hospitals_to_load["A"] = config.HOSPITAL_A_DIR
    if hospital in ("B", "both"):
        hospitals_to_load["B"] = config.HOSPITAL_B_DIR

    for label, directory in hospitals_to_load.items():
        df = _load_psv_directory(directory, max_patients, label)
        if df is not None:
            n_patients = df["patient_id"].nunique()
            meta[label] = n_patients
            frames.append(df)
            print(f"  [OK] Hospital {label}: loaded {n_patients} patients "
                  f"({len(df):,} timestep rows) from {directory}")
        else:
            print(f"  [MISSING] Hospital {label} — no data found at: {directory}")

    if not frames:
        return None, meta

    return pd.concat(frames, ignore_index=True), meta


def _print_download_instructions() -> None:
    """Print dataset download instructions to stdout."""
    msg = textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║     PhysioNet / CinC 2019 Dataset  ─  Not Found             ║
    ╚══════════════════════════════════════════════════════════════╝

    No PSV files were found in either hospital directory:
      Hospital A : {config.HOSPITAL_A_DIR}
      Hospital B : {config.HOSPITAL_B_DIR}

    To obtain the data:
      1.  Register at   https://physionet.org/register/
      2.  Complete CITI training (required for credentialed access)
      3.  Access the challenge page:
            https://physionet.org/content/challenge-2019/1.0.0/
      4.  Download training_setA.zip  and  training_setB.zip
      5.  Extract so the structure is:
            data/raw/training_setA/p000001.psv
            data/raw/training_setA/p000002.psv  ...
            data/raw/training_setB/p100001.psv  ...

    Alternatively, run:
        python scripts/download_data.py --info

    ══════════════════════════════════════════════════════════════
    """)
    print(msg)


# ===========================================================================
# Derived Feature Helpers
# ===========================================================================

def _patient_level_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse row-level data to one row per patient by taking the last value
    of static/demographic features and the *max* SepsisLabel over the stay.

    Returns a patient-level DataFrame with columns:
      ``patient_id``, ``hospital``, ``sepsis_patient``, ``icu_los``,
      ``onset_hour`` (NaN for non-sepsis), ``age``, ``gender``.
    """
    records = []
    for pid, grp in df.groupby("patient_id", sort=False):
        sepsis_any  = int(grp[config.TARGET_COLUMN].max())
        icu_los     = len(grp)

        # Position of the first sepsis label within the patient's sequence
        grp_reset = grp.reset_index(drop=True)
        onset_rows = grp_reset.index[grp_reset[config.TARGET_COLUMN] == 1].tolist()
        onset_hour = float(onset_rows[0] + 1) if (onset_rows and sepsis_any) else float("nan")

        rec = {
            "patient_id"    : pid,
            "hospital"      : grp["hospital"].iloc[0],
            "sepsis_patient": sepsis_any,
            "icu_los"       : icu_los,
            "onset_hour"    : onset_hour,
        }
        # Carry forward static demographics
        for col in ["Age", "Gender"]:
            if col in grp.columns:
                valid = grp[col].dropna()
                rec[col] = float(valid.iloc[-1]) if not valid.empty else float("nan")
        records.append(rec)

    return pd.DataFrame(records)


def _missingness_rates(df: pd.DataFrame) -> pd.Series:
    """Return per-feature missingness rate (0–100 %) for clinical features."""
    feat_cols = [c for c in config.ALL_FEATURES if c in df.columns]
    return df[feat_cols].isna().mean() * 100


# ===========================================================================
# Plot Generators
# ===========================================================================

def _save_fig(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    """Save *fig* to *output_dir*/*filename* and close it."""
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    try:
        rel = out_path.relative_to(Path.cwd())
    except ValueError:
        rel = out_path
    print(f"    [SAVED] {rel}")
    return out_path


# ---------------------------------------------------------------------------
# 1. Class Imbalance
# ---------------------------------------------------------------------------

def plot_class_imbalance(
    df: pd.DataFrame,
    patient_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Two-panel bar chart showing:
      - Left  : sepsis vs non-sepsis *patients*
      - Right : sepsis vs non-sepsis *time windows* (hourly rows)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Imbalance — Patients & Time Windows",
                 fontsize=15, fontweight="bold", y=1.01)

    # Patient-level
    patient_counts = patient_df["sepsis_patient"].value_counts().sort_index()
    labels = ["Non-Sepsis", "Sepsis"]
    colors = PALETTE_BOTH
    p_vals  = [patient_counts.get(0, 0), patient_counts.get(1, 0)]

    axes[0].bar(labels, p_vals, color=colors, edgecolor="white",
                linewidth=0.6, width=0.5)
    axes[0].set_title("Patient-Level", fontsize=12, pad=8)
    axes[0].set_ylabel("Number of Patients", fontsize=10)
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, count in zip(axes[0].patches, p_vals):
        pct = count / sum(p_vals) * 100 if sum(p_vals) else 0
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(p_vals) * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, color="white")

    # Window-level (row-level)
    window_counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    w_vals = [window_counts.get(0, 0), window_counts.get(1, 0)]
    axes[1].bar(labels, w_vals, color=colors, edgecolor="white",
                linewidth=0.6, width=0.5)
    axes[1].set_title("Hourly Time-Window Level", fontsize=12, pad=8)
    axes[1].set_ylabel("Number of Time Windows", fontsize=10)
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, count in zip(axes[1].patches, w_vals):
        pct = count / sum(w_vals) * 100 if sum(w_vals) else 0
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(w_vals) * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, color="white")

    # Imbalance ratio annotation
    if p_vals[1] > 0:
        ratio = p_vals[0] / p_vals[1]
        fig.text(0.5, -0.04,
                 f"Patient-level imbalance ratio  ≈  {ratio:.1f} : 1"
                 "  (non-sepsis : sepsis)",
                 ha="center", fontsize=10, color="#AAAAAA")

    plt.tight_layout()
    _save_fig(fig, output_dir, f"01_class_imbalance.{FIGURE_EXT}")


# ---------------------------------------------------------------------------
# 2. Missing Data Heatmap
# ---------------------------------------------------------------------------

def plot_missing_data_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Horizontal bar chart (sorted by missingness %) for every clinical feature,
    colour-coded by feature category (vital / lab / demographic).
    """
    miss = _missingness_rates(df).sort_values(ascending=False)

    # Category colour map
    cat_colors: List[str] = []
    for feat in miss.index:
        if feat in config.VITAL_SIGNS:
            cat_colors.append("#4FC3F7")   # Blue  – vital
        elif feat in config.LAB_VALUES:
            cat_colors.append("#FFB74D")   # Orange – lab
        else:
            cat_colors.append("#81C784")   # Green  – demographic

    fig, ax = plt.subplots(figsize=(10, max(6, len(miss) * 0.28)))
    bars = ax.barh(miss.index, miss.values, color=cat_colors,
                   edgecolor="none", height=0.7)

    ax.set_xlabel("Missing Data (%)", fontsize=11)
    ax.set_title("Feature Missingness Rates",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.axvline(50, color="white", linestyle="--", linewidth=0.7, alpha=0.5,
               label="50% threshold")

    # Value labels
    for bar, val in zip(bars, miss.values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=7.5, color="white")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4FC3F7", label="Vital Signs"),
        Patch(facecolor="#FFB74D", label="Lab Values"),
        Patch(facecolor="#81C784", label="Demographics"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.3)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"02_missing_data.{FIGURE_EXT}")


# ---------------------------------------------------------------------------
# 3. Vital Signs Distribution Boxplots
# ---------------------------------------------------------------------------

def plot_vital_signs_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Grid of boxplots for each vital sign, split by SepsisLabel (0/1).
    Outliers are capped at the 1st / 99th percentile for readability.
    """
    vitals = [v for v in config.VITAL_SIGNS if v in df.columns]
    n_cols = 4
    n_rows = int(np.ceil(len(vitals) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.5))
    fig.suptitle("Vital Signs Distribution  |  Non-Sepsis vs Sepsis",
                 fontsize=14, fontweight="bold", y=1.01)
    axes_flat = np.array(axes).flatten()

    plot_df = df[[config.TARGET_COLUMN] + vitals].copy()

    for ax, vital in zip(axes_flat, vitals):
        # Cap outliers
        p1, p99 = plot_df[vital].quantile([0.01, 0.99])
        clipped = plot_df[vital].clip(p1, p99)
        temp = pd.DataFrame({vital: clipped,
                             "label": plot_df[config.TARGET_COLUMN]})

        groups = [
            temp.loc[temp["label"] == 0, vital].dropna().values,
            temp.loc[temp["label"] == 1, vital].dropna().values,
        ]
        bp = ax.boxplot(
            groups, patch_artist=True, widths=0.5,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="#AAAAAA"),
            capprops=dict(color="#AAAAAA"),
            flierprops=dict(marker=".", color="#777777",
                            markersize=2, alpha=0.3),
            boxprops=dict(linewidth=0.8),
        )
        for patch, color in zip(bp["boxes"], PALETTE_BOTH):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(vital, fontsize=10, fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Non-Sepsis", "Sepsis"], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

        # Clinical threshold lines
        if vital in config.CLINICAL_THRESHOLDS:
            thr = config.CLINICAL_THRESHOLDS[vital]
            for key, ls in [("low", "--"), ("high", ":")]:
                if thr.get(key) is not None:
                    ax.axhline(thr[key], color="#FFEE58", linestyle=ls,
                               linewidth=0.8, alpha=0.6)

    # Hide unused axes
    for ax in axes_flat[len(vitals):]:
        ax.set_visible(False)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"03_vital_signs_boxplots.{FIGURE_EXT}")


# ---------------------------------------------------------------------------
# 4. Key Lab Values Boxplots
# ---------------------------------------------------------------------------

def plot_lab_values_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Boxplots for a curated set of clinically important lab values
    (Lactate, Creatinine, WBC, BUN, pH), stratified by sepsis label.
    """
    available_labs = [lab for lab in KEY_LABS if lab in df.columns]
    if not available_labs:
        print("  [SKIP] No key lab values found in the loaded data.")
        return

    n_cols = min(len(available_labs), 3)
    n_rows = int(np.ceil(len(available_labs) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.5, n_rows * 4))
    fig.suptitle("Key Lab Values  |  Non-Sepsis vs Sepsis",
                 fontsize=14, fontweight="bold", y=1.01)
    axes_flat = np.array(axes).flatten()

    plot_df = df[[config.TARGET_COLUMN] + available_labs].copy()

    for ax, lab in zip(axes_flat, available_labs):
        p1, p99 = plot_df[lab].quantile([0.01, 0.99])
        clipped = plot_df[lab].clip(p1, p99)
        temp = pd.DataFrame({lab: clipped,
                             "label": plot_df[config.TARGET_COLUMN]})

        groups = [
            temp.loc[temp["label"] == 0, lab].dropna().values,
            temp.loc[temp["label"] == 1, lab].dropna().values,
        ]
        bp = ax.boxplot(
            groups, patch_artist=True, widths=0.5,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color="#AAAAAA"),
            capprops=dict(color="#AAAAAA"),
            flierprops=dict(marker=".", color="#777777",
                            markersize=2, alpha=0.3),
            boxprops=dict(linewidth=0.8),
        )
        for patch, color in zip(bp["boxes"], PALETTE_BOTH):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Median annotations
        for pos, data in enumerate(groups, start=1):
            if len(data):
                median_val = np.median(data)
                ax.text(pos, median_val, f" {median_val:.2f}",
                        va="center", fontsize=8, color="white",
                        fontweight="bold")

        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Non-Sepsis", "Sepsis"], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

        # Clinical thresholds
        if lab in config.CLINICAL_THRESHOLDS:
            thr = config.CLINICAL_THRESHOLDS[lab]
            for key, ls in [("low", "--"), ("high", ":")]:
                if thr.get(key) is not None:
                    ax.axhline(thr[key], color="#FFEE58", linestyle=ls,
                               linewidth=0.9, alpha=0.7,
                               label=f"Normal {key}: {thr[key]}")
            ax.legend(fontsize=7, framealpha=0.25)

    for ax in axes_flat[len(available_labs):]:
        ax.set_visible(False)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"04_lab_values_boxplots.{FIGURE_EXT}")


# ---------------------------------------------------------------------------
# 5. Temporal Sepsis Onset Distribution
# ---------------------------------------------------------------------------

def plot_sepsis_onset(patient_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Histogram of the ICU hour at which sepsis is first recorded,
    limited to patients who develop sepsis.
    """
    onset_data = patient_df["onset_hour"].dropna()

    if onset_data.empty:
        print("  [SKIP] No sepsis onsets found in loaded patients.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(onset_data, bins=min(50, int(onset_data.max())),
            color=PALETTE_SEPSIS, edgecolor="none", alpha=0.85)

    ax.axvline(onset_data.median(), color="#FFEE58", linestyle="--",
               linewidth=1.5,
               label=f"Median onset: {onset_data.median():.1f} h")
    ax.axvline(onset_data.mean(), color="#80DEEA", linestyle=":",
               linewidth=1.5,
               label=f"Mean onset:   {onset_data.mean():.1f} h")

    ax.set_xlabel("ICU Hour at Sepsis Onset", fontsize=12)
    ax.set_ylabel("Number of Patients", fontsize=12)
    ax.set_title("Temporal Distribution of Sepsis Onset in ICU Stay",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Early vs late annotation
    early = int((onset_data < 24).sum())
    late  = int((onset_data >= 24).sum())
    total = len(onset_data)
    ax.text(0.97, 0.95,
            f"Early (< 24 h): {early} ({early/total*100:.1f}%)\n"
            f"Late  (\u226524 h): {late} ({late/total*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="white",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#333333", alpha=0.7))

    plt.tight_layout()
    _save_fig(fig, output_dir, f"05_sepsis_onset_distribution.{FIGURE_EXT}")


# ---------------------------------------------------------------------------
# 6. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Pearson correlation heatmap of vital signs + key lab values.
    Uses a random sample of up to 20,000 rows to keep computation fast.
    """
    corr_features = [f for f in config.VITAL_SIGNS + KEY_LABS
                     if f in df.columns]

    sample_df = df[corr_features].dropna(how="all")
    if len(sample_df) > 20_000:
        sample_df = sample_df.sample(20_000, random_state=42)

    corr_matrix = sample_df.corr(method="pearson")

    # Lower-triangle mask
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr_matrix,
        ax=ax,
        mask=mask,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        linecolor="#333333",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        square=True,
    )
    ax.set_title(
        "Clinical Feature Correlation Heatmap\n"
        "(Vital Signs + Key Lab Values)",
        fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"06_correlation_heatmap.{FIGURE_EXT}")


# ---------------------------------------------------------------------------
# 7. ICU Length-of-Stay Distribution
# ---------------------------------------------------------------------------

def plot_icu_los(patient_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Overlapping histograms of ICU length-of-stay (hours) for sepsis vs
    non-sepsis patients.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    los_neg = patient_df.loc[patient_df["sepsis_patient"] == 0, "icu_los"]
    los_pos = patient_df.loc[patient_df["sepsis_patient"] == 1, "icu_los"]

    max_los = int(patient_df["icu_los"].quantile(0.99))
    bins = np.linspace(0, max_los, 60)

    ax.hist(los_neg, bins=bins, color=PALETTE_NORMAL, alpha=0.6,
            label=f"Non-Sepsis (n={len(los_neg):,})", density=True)
    ax.hist(los_pos, bins=bins, color=PALETTE_SEPSIS, alpha=0.7,
            label=f"Sepsis     (n={len(los_pos):,})", density=True)

    ax.axvline(los_neg.median(), color=PALETTE_NORMAL, linestyle="--",
               linewidth=1.5, alpha=0.9,
               label=f"Median non-sepsis: {los_neg.median():.0f} h")
    ax.axvline(los_pos.median(), color=PALETTE_SEPSIS, linestyle="--",
               linewidth=1.5, alpha=0.9,
               label=f"Median sepsis:     {los_pos.median():.0f} h")

    ax.set_xlabel("ICU Length of Stay (hours)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "ICU Length-of-Stay Distribution  |  Sepsis vs Non-Sepsis",
        fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"07_icu_los_distribution.{FIGURE_EXT}")


# ---------------------------------------------------------------------------
# 8. Hospital Comparison
# ---------------------------------------------------------------------------

def plot_hospital_comparison(
    df: pd.DataFrame,
    patient_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Side-by-side bar chart comparing Hospital A and Hospital B across:
    sepsis rate, mean ICU LOS, and top-3 vital sign mean values.
    Only rendered when both hospitals are present in the data.
    """
    hospitals_present = df["hospital"].unique()
    if len(hospitals_present) < 2:
        print("  [SKIP] Hospital comparison requires data from both A and B.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Hospital A vs Hospital B — Dataset Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    colors     = {"A": "#4FC3F7", "B": "#FF8A65"}
    hosp_labels = ["A", "B"]

    # -- Panel 1: Sepsis rate --------------------------------------------------
    ax = axes[0]
    rates = [
        patient_df.loc[patient_df["hospital"] == h,
                       "sepsis_patient"].mean() * 100
        for h in hosp_labels
    ]
    bars = ax.bar(hosp_labels, rates,
                  color=[colors[h] for h in hosp_labels],
                  edgecolor="white", linewidth=0.5, width=0.4)
    ax.set_title("Sepsis Rate (%)", fontsize=11)
    ax.set_ylabel("% Patients with Sepsis", fontsize=10)
    ax.set_ylim(0, max(rates) * 1.3 if max(rates) else 10)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{rate:.1f}%",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="white")

    # -- Panel 2: Mean ICU LOS ------------------------------------------------
    ax = axes[1]
    mean_los = [
        patient_df.loc[patient_df["hospital"] == h, "icu_los"].mean()
        for h in hosp_labels
    ]
    bars = ax.bar(hosp_labels, mean_los,
                  color=[colors[h] for h in hosp_labels],
                  edgecolor="white", linewidth=0.5, width=0.4)
    ax.set_title("Mean ICU Length of Stay", fontsize=11)
    ax.set_ylabel("Hours", fontsize=10)
    for bar, val in zip(bars, mean_los):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f} h",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="white")

    # -- Panel 3: Mean vital signs comparison ---------------------------------
    ax = axes[2]
    compare_vitals = [v for v in ["HR", "MAP", "Resp"] if v in df.columns]
    x     = np.arange(len(compare_vitals))
    width = 0.35

    for i, hosp in enumerate(hosp_labels):
        means  = [df.loc[df["hospital"] == hosp, v].mean()
                  for v in compare_vitals]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width,
               color=colors[hosp], edgecolor="white",
               linewidth=0.5, label=f"Hospital {hosp}", alpha=0.85)

    ax.set_title("Mean Vital Signs", fontsize=11)
    ax.set_ylabel("Mean Value", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(compare_vitals, fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save_fig(fig, output_dir, f"08_hospital_comparison.{FIGURE_EXT}")


# ===========================================================================
# Text Summary & CSV Export
# ===========================================================================

def print_summary_table(
    df: pd.DataFrame,
    patient_df: pd.DataFrame,
    meta: Dict[str, int],
) -> None:
    """Print a formatted text summary of the loaded dataset."""
    sep_patients    = int(patient_df["sepsis_patient"].sum())
    nonsep_patients = len(patient_df) - sep_patients
    total_patients  = len(patient_df)
    sep_rate        = sep_patients / total_patients * 100 if total_patients else 0

    sep_rows    = int((df[config.TARGET_COLUMN] == 1).sum())
    total_rows  = len(df)
    sep_row_pct = sep_rows / total_rows * 100 if total_rows else 0

    miss = _missingness_rates(df)

    print("\n" + "=" * 66)
    print("  EXPLORATORY DATA ANALYSIS — SUMMARY REPORT")
    print("=" * 66)

    print(f"\n  Dataset Overview")
    print(f"  {'─'*40}")
    print(f"  Hospital A patients loaded : {meta.get('A', 0):>8,}")
    print(f"  Hospital B patients loaded : {meta.get('B', 0):>8,}")
    print(f"  Total unique patients      : {total_patients:>8,}")
    print(f"  Total hourly rows          : {total_rows:>8,}")
    if total_patients:
        print(f"  Avg rows per patient       : {total_rows/total_patients:>8.1f}")

    print(f"\n  Class Distribution")
    print(f"  {'─'*40}")
    print(f"  Sepsis patients   : {sep_patients:>6,}  ({sep_rate:.1f}%)")
    print(f"  Non-Sepsis patts  : {nonsep_patients:>6,}  ({100-sep_rate:.1f}%)")
    print(f"  Sepsis rows       : {sep_rows:>6,}  ({sep_row_pct:.1f}%)")
    if sep_patients:
        print(f"  Imbalance ratio   :  {nonsep_patients/sep_patients:.1f} : 1")
    else:
        print("  Imbalance ratio   : N/A (no sepsis patients found)")

    print(f"\n  ICU Length of Stay  (hours)")
    print(f"  {'─'*40}")
    los = patient_df["icu_los"]
    print(f"  Min / Max          : {los.min():.0f} / {los.max():.0f}")
    print(f"  Mean \u00b1 Std         : {los.mean():.1f} \u00b1 {los.std():.1f}")
    print(f"  Median (IQR)       : {los.median():.0f} "
          f"({los.quantile(.25):.0f}\u2013{los.quantile(.75):.0f})")

    print(f"\n  Sepsis Onset Hour  (among sepsis patients)")
    print(f"  {'─'*40}")
    onset = patient_df["onset_hour"].dropna()
    if not onset.empty:
        print(f"  Mean \u00b1 Std         : {onset.mean():.1f} \u00b1 {onset.std():.1f}")
        print(f"  Median             : {onset.median():.0f}")
        print(f"  Early onset (< 24 h): "
              f"{int((onset < 24).sum())} ({(onset < 24).mean()*100:.1f}%)")
    else:
        print("  No sepsis onset data available.")

    print(f"\n  Top 10 Features by Missingness (%)")
    print(f"  {'─'*40}")
    for feat, pct in miss.sort_values(ascending=False).head(10).items():
        bar_str = "\u2588" * int(pct / 5)
        print(f"  {feat:<20} {pct:5.1f}%  {bar_str}")

    print(f"\n  Vital Signs Statistics (row-level, all patients)")
    print(f"  {'─'*50}")
    print(f"  {'Feature':<12} {'Mean':>8} {'Std':>8} "
          f"{'Median':>8} {'Miss%':>7}")
    print(f"  {'─'*50}")
    for vs in config.VITAL_SIGNS:
        if vs not in df.columns:
            continue
        col = df[vs]
        print(f"  {vs:<12} {col.mean():>8.2f} {col.std():>8.2f} "
              f"{col.median():>8.2f} {miss.get(vs, 0):>6.1f}%")

    print("=" * 66 + "\n")


def save_summary_csv(
    df: pd.DataFrame,
    patient_df: pd.DataFrame,
) -> Path:
    """
    Save a per-feature statistics CSV to
    ``config.TABLES_DIR / eda_summary.csv``.

    Columns: feature, category, mean, std, median, q25, q75,
             min, max, missing_pct, sepsis_mean, nonsepsis_mean,
             n_total_rows.
    """
    all_feat = [f for f in config.ALL_FEATURES if f in df.columns]
    records  = []

    for feat in all_feat:
        if feat in config.VITAL_SIGNS:
            cat = "vital_sign"
        elif feat in config.LAB_VALUES:
            cat = "lab_value"
        else:
            cat = "demographic"

        col     = df[feat]
        col_sep = df.loc[df[config.TARGET_COLUMN] == 1, feat]
        col_neg = df.loc[df[config.TARGET_COLUMN] == 0, feat]

        records.append({
            "feature"       : feat,
            "category"      : cat,
            "mean"          : col.mean(),
            "std"           : col.std(),
            "median"        : col.median(),
            "q25"           : col.quantile(0.25),
            "q75"           : col.quantile(0.75),
            "min"           : col.min(),
            "max"           : col.max(),
            "missing_pct"   : col.isna().mean() * 100,
            "sepsis_mean"   : col_sep.mean(),
            "nonsepsis_mean": col_neg.mean(),
            "n_total_rows"  : int(col.notna().sum()),
        })

    summary_df = pd.DataFrame(records)
    out_path   = config.TABLES_DIR / "eda_summary.csv"
    summary_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  [SAVED] Summary CSV \u2192 {out_path}")
    return out_path


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="eda.py",
        description=(
            "Exploratory Data Analysis for the PhysioNet/CinC 2019 "
            "Sepsis Early Prediction dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          python scripts/eda.py --hospital A --max-patients 500
          python scripts/eda.py --hospital both --output-dir results/figures/eda
          python scripts/eda.py --hospital B --max-patients 1000 --output-dir custom/path
        """),
    )
    parser.add_argument(
        "--hospital",
        choices=["A", "B", "both"],
        default="both",
        help="Which hospital dataset to analyse. (default: both)",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=500,
        metavar="N",
        help="Maximum number of patients to load per hospital. (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.FIGURES_DIR / "eda",
        metavar="PATH",
        help="Directory to save generated figures. "
             "(default: results/figures/eda)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only print the text summary; do not generate any plots.",
    )
    return parser


def main() -> int:
    """
    Main entry point.

    Returns
    -------
    Exit code: 0 on success, 1 on fatal error.
    """
    parser = build_arg_parser()
    args   = parser.parse_args()

    print("\n" + "=" * 66)
    print("  PhysioNet/CinC 2019 — Exploratory Data Analysis")
    print("=" * 66)
    print(f"\n  Config:")
    print(f"    Hospital(s)  : {args.hospital}")
    print(f"    Max patients : {args.max_patients:,} per hospital")
    print(f"    Output dir   : {args.output_dir}")

    # ── Create output directory ────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n  Loading data ...")
    df, meta = load_data(hospital=args.hospital,
                         max_patients=args.max_patients)

    if df is None or df.empty:
        _print_download_instructions()
        return 1

    print(f"\n  Loaded {len(df):,} rows across "
          f"{df['patient_id'].nunique():,} patients.\n")

    # ── Validate target column ─────────────────────────────────────────────
    if config.TARGET_COLUMN not in df.columns:
        print(f"  [ERROR] Column '{config.TARGET_COLUMN}' not found in data. "
              "Check PSV file format.")
        return 1

    # ── Patient-level aggregation ─────────────────────────────────────────
    print("  Building patient-level summary ...")
    patient_df = _patient_level_df(df)

    # ── Text summary ──────────────────────────────────────────────────────
    print_summary_table(df, patient_df, meta)

    # ── Save summary CSV ──────────────────────────────────────────────────
    print("  Saving summary CSV ...")
    save_summary_csv(df, patient_df)

    if args.skip_plots:
        print("\n  [INFO] --skip-plots flag set; skipping figure generation.")
        return 0

    # ── Generate plots ────────────────────────────────────────────────────
    print(f"\n  Generating plots -> {args.output_dir}\n")

    steps = [
        ("Class imbalance chart",
         lambda: plot_class_imbalance(df, patient_df, args.output_dir)),
        ("Missing data heatmap",
         lambda: plot_missing_data_heatmap(df, args.output_dir)),
        ("Vital signs boxplots",
         lambda: plot_vital_signs_boxplots(df, args.output_dir)),
        ("Key lab values boxplots",
         lambda: plot_lab_values_boxplots(df, args.output_dir)),
        ("Sepsis onset distribution",
         lambda: plot_sepsis_onset(patient_df, args.output_dir)),
        ("Correlation heatmap",
         lambda: plot_correlation_heatmap(df, args.output_dir)),
        ("ICU length-of-stay distribution",
         lambda: plot_icu_los(patient_df, args.output_dir)),
        ("Hospital comparison",
         lambda: plot_hospital_comparison(df, patient_df, args.output_dir)),
    ]

    for i, (label, func) in enumerate(steps, start=1):
        print(f"  [{i}/{len(steps)}] {label} ...")
        try:
            func()
        except Exception as exc:  # noqa: BLE001
            print(f"    [WARN] Failed to generate '{label}': {exc}")

    # ── Final summary ─────────────────────────────────────────────────────
    saved_files = sorted(args.output_dir.glob(f"*.{FIGURE_EXT}"))
    print(f"\n  Done.  {len(saved_files)} figure(s) saved to:")
    print(f"     {args.output_dir}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
