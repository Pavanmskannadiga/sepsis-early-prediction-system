"""
Data Preprocessing Module for the Sepsis Early Prediction System.

Handles loading PSV files from the PhysioNet/CinC 2019 Challenge dataset,
feature extraction, missing value imputation, normalization, and sequence
creation for BiLSTM input.

The PhysioNet/CinC 2019 dataset stores each patient as a pipe-delimited
text file (.psv) with hourly rows and 41 columns (40 features + SepsisLabel).
The 6-hour prediction horizon is already encoded in the SepsisLabel column.

References:
    - Reyna et al., "Early prediction of sepsis from clinical data: The
      PhysioNet Challenge," Critical Care Medicine, 2020.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def load_single_psv(filepath: str) -> pd.DataFrame:
    """
    Load a single patient PSV file.

    Each PSV file is pipe-delimited with columns for vital signs,
    lab values, demographics, and the SepsisLabel target.

    Args:
        filepath: Path to a .psv file.

    Returns:
        DataFrame with all 41 columns, one row per hour.
    """
    return pd.read_csv(filepath, sep="|")


def load_psv_files(
    directory: str,
    max_patients: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Load all patient PSV files from a directory.

    Args:
        directory: Path to directory containing .psv files.
        max_patients: If set, limit to this many patients (for debugging).
        verbose: Show progress bar.

    Returns:
        Tuple of (list of DataFrames, list of patient IDs).
    """
    directory = Path(directory)
    psv_files = sorted(directory.glob("*.psv"))

    if max_patients is not None:
        psv_files = psv_files[:max_patients]

    patients = []
    patient_ids = []

    iterator = tqdm(psv_files, desc=f"Loading {directory.name}") if verbose else psv_files

    for filepath in iterator:
        try:
            df = load_single_psv(filepath)
            patients.append(df)
            patient_ids.append(filepath.stem)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load {filepath.name}: {e}")

    if verbose:
        print(f"Loaded {len(patients)} patients from {directory.name}")

    return patients, patient_ids


def extract_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the 34 clinical features (8 vitals + 26 labs) from a patient
    DataFrame, excluding demographics which are static.

    Args:
        df: Patient DataFrame with all 41 columns.

    Returns:
        DataFrame with only the 34 clinical feature columns.
    """
    available = [f for f in config.CLINICAL_FEATURES if f in df.columns]
    return df[available].copy()


def create_missingness_mask(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary mask indicating which values are observed (1) vs
    missing (0). Missingness itself is clinically informative — stable
    patients are tested less frequently, so a missing lab value carries
    information about the patient's perceived stability.

    Args:
        df: DataFrame with potential NaN values.

    Returns:
        DataFrame of same shape with 1 for observed, 0 for missing.
    """
    return (~df.isna()).astype(np.float32)


def impute_missing_values(
    df: pd.DataFrame,
    strategy: str = "forward_fill",
    population_medians: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Impute missing values using a multi-step strategy:
      1. Forward-fill (carry last observation forward)
      2. Backward-fill (fill initial NaNs with first available value)
      3. Fill remaining with population medians

    This is the most common imputation strategy in the sepsis-prediction
    literature and is clinically intuitive: the most recent measurement
    is the best estimate of the current value.

    Args:
        df: DataFrame with missing values.
        strategy: Imputation strategy ("forward_fill", "median", "zero").
        population_medians: Dict of feature→median for final fill.

    Returns:
        DataFrame with no NaN values.
    """
    if population_medians is None:
        population_medians = config.POPULATION_MEDIANS

    df = df.copy()

    if strategy == "forward_fill":
        # Step 1: Forward fill
        df = df.ffill()
        # Step 2: Backward fill for initial NaN values
        df = df.bfill()
    elif strategy == "median":
        pass  # Will be filled by population medians below
    elif strategy == "zero":
        df = df.fillna(0.0)
        return df

    # Step 3: Fill any remaining NaN with population medians
    for col in df.columns:
        if df[col].isna().any():
            median_val = population_medians.get(col, 0.0)
            df[col] = df[col].fillna(median_val)

    return df


def normalize_features(
    train_data: np.ndarray,
    val_data: Optional[np.ndarray] = None,
    test_data: Optional[np.ndarray] = None,
    method: str = "standard"
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], object]:
    """
    Normalize features using statistics computed ONLY from training data.
    This prevents data leakage from validation/test sets.

    Args:
        train_data: Training features, shape (n_samples, n_features) or
                    (n_samples, seq_len, n_features).
        val_data: Optional validation features.
        test_data: Optional test features.
        method: "standard" (z-score) or "minmax".

    Returns:
        Tuple of (normalized_train, normalized_val, normalized_test, scaler).
    """
    original_shape = train_data.shape

    # Reshape to 2D for sklearn scalers
    if train_data.ndim == 3:
        n_samples, seq_len, n_features = train_data.shape
        train_2d = train_data.reshape(-1, n_features)
        val_2d = val_data.reshape(-1, n_features) if val_data is not None else None
        test_2d = test_data.reshape(-1, n_features) if test_data is not None else None
    else:
        train_2d = train_data
        val_2d = val_data
        test_2d = test_data
        n_features = train_data.shape[-1]

    # Fit scaler on training data only
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    train_scaled = scaler.fit_transform(train_2d)

    val_scaled = scaler.transform(val_2d) if val_2d is not None else None
    test_scaled = scaler.transform(test_2d) if test_2d is not None else None

    # Reshape back to 3D if needed
    if train_data.ndim == 3:
        train_scaled = train_scaled.reshape(original_shape)
        if val_scaled is not None:
            val_scaled = val_scaled.reshape(val_data.shape)
        if test_scaled is not None:
            test_scaled = test_scaled.reshape(test_data.shape)

    return train_scaled, val_scaled, test_scaled, scaler


def create_sequences(
    patient_data: pd.DataFrame,
    patient_labels: np.ndarray,
    seq_length: int = 48,
    step_size: int = 1
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Create fixed-length sequences from a patient's time series using
    a sliding window approach. Each sequence is paired with the label
    at its last timestep.

    The PhysioNet dataset already encodes the 6-hour look-ahead in the
    SepsisLabel column, so the label at position t indicates whether
    sepsis will occur within 6 hours of time t.

    Args:
        patient_data: Feature array, shape (T, F).
        patient_labels: Label array, shape (T,).
        seq_length: Length of each sequence window (hours).
        step_size: Step size for the sliding window.

    Returns:
        Tuple of (list of sequence arrays, list of labels).
    """
    T = len(patient_data)
    sequences = []
    labels = []

    if T < config.MIN_SEQUENCE_LENGTH:
        return sequences, labels

    if T <= seq_length:
        # Pad shorter sequences with zeros at the beginning
        pad_length = seq_length - T
        data_array = patient_data.values if isinstance(patient_data, pd.DataFrame) else patient_data
        padded = np.pad(data_array, ((pad_length, 0), (0, 0)), mode="constant", constant_values=0)
        sequences.append(padded)
        labels.append(int(patient_labels[-1]))
    else:
        data_array = patient_data.values if isinstance(patient_data, pd.DataFrame) else patient_data
        for start in range(0, T - seq_length + 1, step_size):
            end = start + seq_length
            seq = data_array[start:end]
            label = int(patient_labels[end - 1])
            sequences.append(seq)
            labels.append(label)

    return sequences, labels


def process_hospital_data(
    directory: str,
    max_patients: Optional[int] = None,
    use_mask: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Full preprocessing pipeline for one hospital's data:
      1. Load all PSV files
      2. Extract clinical features
      3. Create missingness masks
      4. Impute missing values
      5. Concatenate features and masks
      6. Create sequences

    Args:
        directory: Path to hospital data directory.
        max_patients: Limit number of patients (for debugging).
        use_mask: Whether to include missingness masks as features.
        verbose: Show progress.

    Returns:
        Tuple of (X sequences array, y labels array, patient_ids).
    """
    patients, patient_ids = load_psv_files(directory, max_patients, verbose)

    all_sequences = []
    all_labels = []
    processed_ids = []

    desc = "Processing patients"
    iterator = tqdm(zip(patients, patient_ids), total=len(patients), desc=desc) if verbose else zip(patients, patient_ids)

    for df, pid in iterator:
        # Extract clinical features
        clinical_df = extract_clinical_features(df)

        # Create missingness mask before imputation
        if use_mask:
            mask_df = create_missingness_mask(clinical_df)

        # Impute missing values
        imputed_df = impute_missing_values(
            clinical_df,
            strategy=config.IMPUTATION_STRATEGY
        )

        # Combine features and masks
        if use_mask:
            # Rename mask columns to avoid collision
            mask_df.columns = [f"{col}_mask" for col in mask_df.columns]
            combined_df = pd.concat([imputed_df, mask_df], axis=1)
        else:
            combined_df = imputed_df

        # Extract labels
        labels = df[config.TARGET_COLUMN].values.astype(np.int32)

        # Create sequences
        seqs, seq_labels = create_sequences(
            combined_df,
            labels,
            seq_length=config.SEQUENCE_LENGTH,
            step_size=1
        )

        if len(seqs) > 0:
            all_sequences.extend(seqs)
            all_labels.extend(seq_labels)
            processed_ids.append(pid)

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    if verbose:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        print(f"\nDataset summary:")
        print(f"  Total sequences: {len(y)}")
        if len(y) > 0:
            print(f"  Positive (sepsis): {n_pos} ({100*n_pos/len(y):.1f}%)")
            print(f"  Negative (no sepsis): {n_neg} ({100*n_neg/len(y):.1f}%)")
        else:
            print(f"  Positive (sepsis): 0 (0.0%)")
            print(f"  Negative (no sepsis): 0 (0.0%)")
        print(f"  Sequence shape: {X.shape}")

    return X, y, processed_ids


def prepare_dataset(
    hospital: str = "A",
    max_patients: Optional[int] = None,
    normalize: bool = True,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Complete data preparation pipeline: load, preprocess, split, normalize.

    Args:
        hospital: "A", "B", or "AB" (both).
        max_patients: Limit patients (for debugging).
        normalize: Whether to normalize features.
        verbose: Show progress.

    Returns:
        Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test,
        and optionally the scaler object.
    """
    if hospital == "A":
        X, y, _ = process_hospital_data(
            str(config.HOSPITAL_A_DIR), max_patients, verbose=verbose
        )
    elif hospital == "B":
        X, y, _ = process_hospital_data(
            str(config.HOSPITAL_B_DIR), max_patients, verbose=verbose
        )
    elif hospital == "AB":
        X_a, y_a, _ = process_hospital_data(
            str(config.HOSPITAL_A_DIR), max_patients, verbose=verbose
        )
        X_b, y_b, _ = process_hospital_data(
            str(config.HOSPITAL_B_DIR), max_patients, verbose=verbose
        )
        X = np.concatenate([X_a, X_b], axis=0)
        y = np.concatenate([y_a, y_b], axis=0)
    else:
        raise ValueError(f"Unknown hospital: {hospital}. Use 'A', 'B', or 'AB'.")

    if len(X) == 0:
        raise ValueError(
            f"No sequence data found. Please ensure the PhysioNet/CinC 2019 dataset .psv files "
            f"are downloaded and extracted in your raw data directories: "
            f"\n  Hospital A: {config.HOSPITAL_A_DIR}"
            f"\n  Hospital B: {config.HOSPITAL_B_DIR}"
            f"\nRun `python scripts/download_data.py --info` for details."
        )

    # Shuffle
    np.random.seed(config.TrainConfig.random_seed)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Split into train/val/test
    n = len(X)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    scaler = None
    if normalize:
        X_train, X_val, X_test, scaler = normalize_features(
            X_train, X_val, X_test, method=config.NORMALIZATION
        )

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    if scaler is not None:
        result["scaler"] = scaler

    if verbose:
        print(f"\nFinal splits:")
        print(f"  Train: {X_train.shape}, pos={y_train.sum()}")
        print(f"  Val:   {X_val.shape}, pos={y_val.sum()}")
        print(f"  Test:  {X_test.shape}, pos={y_test.sum()}")

    return result
