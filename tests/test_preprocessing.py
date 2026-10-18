"""
test_preprocessing.py — Unit tests for the data-preprocessing pipeline.

Covers:
    - Sample DataFrame structure and SepsisLabel column
    - Binary nature of SepsisLabel values
    - Forward-fill reduces (or keeps equal) the NaN count
    - config.CLINICAL_FEATURES contains exactly 34 features
    - config.VITAL_SIGNS contains exactly 8 features
    - config.LAB_VALUES contains exactly 26 features
    - config.DEMOGRAPHICS contains exactly 6 features
"""

import sys
sys.path.insert(0, 'd:/sp system')

import pytest
import numpy as np
import pandas as pd
from io import StringIO

import config


# ---------------------------------------------------------------------------
# Helper: build a minimal sample DataFrame with all required columns
# ---------------------------------------------------------------------------

ALL_COLUMNS = [
    # Vital signs (8)
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    # Lab values (26)
    "Lactate", "Creatinine", "WBC", "pH", "BUN", "HCO3", "FiO2", "PaCO2",
    "SaO2", "AST", "Alkalinephos", "Calcium", "Chloride", "Bilirubin_direct",
    "Glucose", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "Fibrinogen", "Platelets", "BaseExcess",
    # Demographics (6)
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    # Target
    "SepsisLabel",
]


def make_sample_df(n_rows: int = 20, nan_fraction: float = 0.15) -> pd.DataFrame:
    """
    Build a synthetic patient DataFrame that mirrors the PhysioNet PSV schema.

    Args:
        n_rows:        Number of hourly observations to generate.
        nan_fraction:  Fraction of cells (excluding SepsisLabel) to set NaN.

    Returns:
        pd.DataFrame with all required columns, random numeric data,
        binary SepsisLabel, and some strategically inserted NaN values.
    """
    rng = np.random.default_rng(seed=42)

    data = {}
    for col in ALL_COLUMNS:
        if col == "SepsisLabel":
            data[col] = rng.integers(0, 2, size=n_rows).astype(float)
        elif col in ("Gender", "Unit1", "Unit2"):
            data[col] = rng.integers(0, 2, size=n_rows).astype(float)
        else:
            data[col] = rng.uniform(0.1, 200.0, size=n_rows).astype(float)

    df = pd.DataFrame(data)

    # Inject NaN into every clinical column (not SepsisLabel)
    clinical_cols = [c for c in ALL_COLUMNS if c != "SepsisLabel"]
    n_nan = max(1, int(n_rows * len(clinical_cols) * nan_fraction))
    nan_row_idx = rng.integers(0, n_rows, size=n_nan)
    nan_col_idx = rng.integers(0, len(clinical_cols), size=n_nan)
    for r, c in zip(nan_row_idx, nan_col_idx):
        df.iloc[r, df.columns.get_loc(clinical_cols[c])] = np.nan

    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSampleDataFrame:
    """Tests that validate the structure and content of make_sample_df()."""

    def test_dataframe_has_expected_columns(self):
        """The sample DataFrame must contain a 'SepsisLabel' column."""
        df = make_sample_df()
        assert "SepsisLabel" in df.columns, (
            "make_sample_df() must produce a DataFrame with a 'SepsisLabel' column"
        )

    def test_sepsis_label_binary(self):
        """Every value in SepsisLabel must be 0 or 1 (no other values allowed)."""
        df = make_sample_df()
        unique_vals = set(df["SepsisLabel"].dropna().astype(int).unique())
        assert unique_vals.issubset({0, 1}), (
            f"SepsisLabel contains non-binary values: {unique_vals}"
        )


class TestForwardFill:
    """Tests for pandas forward-fill imputation behaviour."""

    def test_forward_fill_reduces_nans(self):
        """
        After pd.DataFrame.ffill(), the total NaN count must be ≤ the
        original count (forward-fill can never introduce new NaNs).
        """
        df = make_sample_df()
        nan_before = df.isna().sum().sum()
        df_filled = df.ffill()
        nan_after  = df_filled.isna().sum().sum()
        assert nan_after <= nan_before, (
            f"ffill() increased NaN count: {nan_before} → {nan_after}"
        )


class TestConfigFeatureCounts:
    """Tests that verify the feature-list sizes in config.py."""

    def test_feature_count(self):
        """config.CLINICAL_FEATURES must contain exactly 34 features."""
        assert len(config.CLINICAL_FEATURES) == 34, (
            f"Expected 34 clinical features, found {len(config.CLINICAL_FEATURES)}"
        )

    def test_vital_signs_count(self):
        """config.VITAL_SIGNS must contain exactly 8 features."""
        assert len(config.VITAL_SIGNS) == 8, (
            f"Expected 8 vital signs, found {len(config.VITAL_SIGNS)}"
        )

    def test_lab_values_count(self):
        """config.LAB_VALUES must contain exactly 26 features."""
        assert len(config.LAB_VALUES) == 26, (
            f"Expected 26 lab values, found {len(config.LAB_VALUES)}"
        )

    def test_demographics_count(self):
        """config.DEMOGRAPHICS must contain exactly 6 features."""
        assert len(config.DEMOGRAPHICS) == 6, (
            f"Expected 6 demographic features, found {len(config.DEMOGRAPHICS)}"
        )
