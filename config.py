"""
Configuration module for the Sepsis Early Prediction System.

Central location for all hyperparameters, file paths, feature definitions,
and training settings. Modify values here rather than in individual modules
to ensure consistency across the pipeline.

Authors: Pavana M.S., Megharaja S., Srujan S. Sagar, Abubakkar Siddiq
Guide: Prof. Shilpa K.C.
Dept. of AIML, PESITM Shivamogga, 2026
"""

import os
from pathlib import Path

# ============================================================
# Directory Paths
# ============================================================
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HOSPITAL_A_DIR = RAW_DATA_DIR / "training_setA"
HOSPITAL_B_DIR = RAW_DATA_DIR / "training_setB"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR = RESULTS_DIR / "tables"

# Create directories if they don't exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
          HOSPITAL_A_DIR, HOSPITAL_B_DIR,
          RESULTS_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# PhysioNet/CinC 2019 Challenge Feature Definitions
# ============================================================

# Vital Signs (8 features)
VITAL_SIGNS = [
    "HR",       # Heart rate (beats per minute)
    "O2Sat",    # Pulse oximetry (%)
    "Temp",     # Temperature (°C)
    "SBP",      # Systolic BP (mm Hg)
    "MAP",      # Mean arterial pressure (mm Hg)
    "DBP",      # Diastolic BP (mm Hg)
    "Resp",     # Respiration rate (breaths per minute)
    "EtCO2",    # End tidal CO2 (mm Hg)
]

# Laboratory Values (26 features)
LAB_VALUES = [
    "BaseExcess",       # Base excess (mmol/L)
    "HCO3",             # Bicarbonate (mmol/L)
    "FiO2",             # Fraction of inspired oxygen (%)
    "pH",               # Arterial pH
    "PaCO2",            # Partial pressure of CO2 (mm Hg)
    "SaO2",             # Arterial oxygen saturation (%)
    "AST",              # Aspartate transaminase (IU/L)
    "BUN",              # Blood urea nitrogen (mg/dL)
    "Alkalinephos",     # Alkaline phosphatase (IU/L)
    "Calcium",          # Calcium (mg/dL)
    "Chloride",         # Chloride (mmol/L)
    "Creatinine",       # Creatinine (mg/dL)
    "Bilirubin_direct", # Direct bilirubin (mg/dL)
    "Glucose",          # Glucose (mg/dL)
    "Lactate",          # Lactate (mmol/L)
    "Magnesium",        # Magnesium (mmol/L)
    "Phosphate",        # Phosphate (mg/dL)
    "Potassium",        # Potassium (mmol/L)
    "Bilirubin_total",  # Total bilirubin (mg/dL)
    "TroponinI",        # Troponin I (ng/mL)
    "Hct",              # Hematocrit (%)
    "Hgb",              # Hemoglobin (g/dL)
    "PTT",              # Partial thromboplastin time (seconds)
    "WBC",              # White blood cell count (10^3/µL)
    "Fibrinogen",       # Fibrinogen (mg/dL)
    "Platelets",        # Platelets (10^3/µL)
]

# Demographic Features (6 features)
DEMOGRAPHICS = [
    "Age",           # Years (capped at 100 for privacy)
    "Gender",        # Female (0), Male (1)
    "Unit1",         # Administrative identifier (MICU)
    "Unit2",         # Administrative identifier (SICU)
    "HospAdmTime",   # Hours between hospital and ICU admission
    "ICULOS",        # ICU length of stay (hours)
]

# Target label
TARGET_COLUMN = "SepsisLabel"

# All clinical features (34 features, excluding demographics that are static)
CLINICAL_FEATURES = VITAL_SIGNS + LAB_VALUES

# All input features including demographics
ALL_FEATURES = VITAL_SIGNS + LAB_VALUES + DEMOGRAPHICS

# Total feature counts
NUM_VITAL_SIGNS = len(VITAL_SIGNS)        # 8
NUM_LAB_VALUES = len(LAB_VALUES)          # 26
NUM_DEMOGRAPHICS = len(DEMOGRAPHICS)       # 6
NUM_CLINICAL_FEATURES = len(CLINICAL_FEATURES)  # 34
NUM_ALL_FEATURES = len(ALL_FEATURES)      # 40

# Features to use for model input (clinical features + missingness masks)
# When using missingness masks, total input = 34 * 2 = 68
USE_MISSINGNESS_MASK = True
NUM_INPUT_FEATURES = NUM_CLINICAL_FEATURES * 2 if USE_MISSINGNESS_MASK else NUM_CLINICAL_FEATURES

# ============================================================
# Data Pipeline Settings
# ============================================================

# Sequence / windowing
SEQUENCE_LENGTH = 48          # Hours of observation window
PREDICTION_HORIZON = 6        # Predict sepsis 6 hours ahead
MIN_SEQUENCE_LENGTH = 8       # Minimum hours of data required

# Train/validation/test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Missing data handling
IMPUTATION_STRATEGY = "forward_fill"  # "forward_fill", "median", "zero"
FILL_REMAINING_WITH = "median"        # After forward fill, fill NaN with median

# Normalization
NORMALIZATION = "standard"  # "standard" (z-score) or "minmax"

# ============================================================
# Model Hyperparameters — BiLSTM + Attention
# ============================================================

class ModelConfig:
    """BiLSTM + Multi-Head Attention model configuration."""

    # LSTM parameters
    input_dim = NUM_INPUT_FEATURES     # 68 with masks, 34 without
    hidden_dim = 128                   # Hidden size per direction
    num_layers = 2                     # Number of BiLSTM layers
    dropout = 0.3                      # Dropout rate

    # Attention parameters
    num_attention_heads = 4            # Multi-head attention heads
    attention_dim = hidden_dim * 2     # 256 (bidirectional output)

    # Fully connected layers
    fc_hidden_dim = 128                # Hidden FC layer size
    output_dim = 1                     # Binary classification

    # Bidirectional flag
    bidirectional = True


# ============================================================
# Model Hyperparameters — Transformer Encoder
# ============================================================

class TransformerConfig:
    """Transformer Encoder model configuration."""

    input_dim = NUM_INPUT_FEATURES     # 68 with masks, 34 without
    d_model = 256                      # Transformer model dimension
    num_heads = 4                      # Number of attention heads
    num_layers = 4                     # Number of encoder layers
    d_ff = 512                         # Feed-forward hidden dimension
    dropout = 0.3                      # Dropout rate
    fc_hidden_dim = 128                # Classification head hidden dim
    output_dim = 1                     # Binary classification
    max_seq_len = 512                  # Max positional encoding length


# ============================================================
# Model Hyperparameters — Temporal Convolutional Network (TCN)
# ============================================================

class TCNConfig:
    """Temporal Convolutional Network model configuration."""

    input_dim = NUM_INPUT_FEATURES     # 68 with masks, 34 without
    num_channels = [128, 128, 128, 128]  # Output channels per TCN block
    kernel_size = 3                    # Convolution kernel size
    dropout = 0.3                      # Dropout rate
    fc_hidden_dim = 128                # Classification head hidden dim
    output_dim = 1                     # Binary classification


# ============================================================
# Training Hyperparameters
# ============================================================

class TrainConfig:
    """Training loop configuration."""

    # Optimization
    learning_rate = 1e-3
    weight_decay = 1e-5                # L2 regularization
    batch_size = 64
    max_epochs = 100
    gradient_clip_norm = 1.0           # Max gradient norm

    # Learning rate schedule
    lr_scheduler = "cosine"            # "cosine", "step", "plateau"
    lr_step_size = 20                  # For StepLR
    lr_gamma = 0.5                     # For StepLR
    lr_min = 1e-6                      # Minimum LR for cosine

    # Early stopping
    patience = 10                      # Epochs without improvement
    min_delta = 1e-4                   # Minimum improvement threshold
    monitor_metric = "val_auroc"       # Metric to monitor

    # Loss function
    loss_function = "focal"            # "bce", "weighted_bce", "focal"
    focal_gamma = 2.0                  # Focal loss gamma
    focal_alpha = 0.75                 # Focal loss alpha (weight for positive class)

    # Reproducibility
    random_seed = 42

    # Device
    device = "auto"  # "auto", "cuda", "cpu"


# ============================================================
# Imbalance Handling Settings
# ============================================================

class ImbalanceConfig:
    """Class imbalance handling configuration."""

    # SMOTE settings
    smote_strategy = "auto"            # Sampling strategy
    smote_k_neighbors = 5             # k for SMOTE
    smote_random_state = 42

    # Alternative: ADASYN
    adasyn_n_neighbors = 5

    # SMOTE-ENN hybrid
    use_smote_enn = False              # SMOTE + Edited Nearest Neighbors

    # Weighted loss (alternative to SMOTE)
    use_class_weights = True           # Use inverse frequency weights
    pos_weight_multiplier = 1.0        # Scale the positive weight


# ============================================================
# SHAP Explainability Settings
# ============================================================

class SHAPConfig:
    """SHAP analysis configuration."""

    # Explainer type
    explainer_type = "deep"            # "deep" (DeepExplainer) or "kernel" (KernelExplainer)

    # Background samples for SHAP
    num_background_samples = 100       # Number of background samples
    use_kmeans_background = True       # Use k-means summary

    # Explanation samples
    num_explain_samples = 200          # Number of test samples to explain

    # Visualization
    top_k_features = 10               # Top features to show
    save_format = "png"               # "png", "pdf", "svg"
    dpi = 300                         # Figure DPI


# ============================================================
# Cross-Hospital Validation Settings
# ============================================================

class CrossHospitalConfig:
    """Cross-hospital generalization validation."""

    # Experiments to run
    experiments = [
        {"train": "A", "test": "A", "name": "Same-Hospital A"},
        {"train": "A", "test": "B", "name": "Cross: A→B"},
        {"train": "B", "test": "A", "name": "Cross: B→A"},
        {"train": "B", "test": "B", "name": "Same-Hospital B"},
        {"train": "AB", "test": "A", "name": "Combined→A"},
        {"train": "AB", "test": "B", "name": "Combined→B"},
    ]

    # Compare with/without SMOTE
    compare_smote = True

    # Number of bootstrap iterations for CI
    n_bootstrap = 100
    bootstrap_ci = 0.95               # 95% confidence interval


# ============================================================
# Clinically Relevant Thresholds (for reference)
# ============================================================

CLINICAL_THRESHOLDS = {
    "HR": {"low": 60, "high": 100, "unit": "bpm"},
    "SBP": {"low": 90, "high": 140, "unit": "mmHg"},
    "MAP": {"low": 65, "high": 110, "unit": "mmHg"},
    "Resp": {"low": 12, "high": 20, "unit": "breaths/min"},
    "Temp": {"low": 36.0, "high": 38.0, "unit": "°C"},
    "O2Sat": {"low": 94, "high": 100, "unit": "%"},
    "Lactate": {"low": 0.5, "high": 2.0, "unit": "mmol/L"},
    "WBC": {"low": 4.0, "high": 12.0, "unit": "10³/µL"},
    "Creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL"},
    "Platelets": {"low": 150, "high": 400, "unit": "10³/µL"},
    "Bilirubin_total": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
}


# ============================================================
# Population Medians (CinC 2019 approximate values for imputation)
# ============================================================

POPULATION_MEDIANS = {
    "HR": 80.0, "O2Sat": 97.0, "Temp": 36.8, "SBP": 120.0,
    "MAP": 80.0, "DBP": 65.0, "Resp": 18.0, "EtCO2": 35.0,
    "BaseExcess": 0.0, "HCO3": 24.0, "FiO2": 0.21, "pH": 7.4,
    "PaCO2": 40.0, "SaO2": 97.0, "AST": 25.0, "BUN": 15.0,
    "Alkalinephos": 70.0, "Calcium": 8.8, "Chloride": 103.0,
    "Creatinine": 1.0, "Bilirubin_direct": 0.2, "Glucose": 120.0,
    "Lactate": 1.5, "Magnesium": 2.0, "Phosphate": 3.5,
    "Potassium": 4.0, "Bilirubin_total": 0.6, "TroponinI": 0.01,
    "Hct": 35.0, "Hgb": 11.5, "PTT": 30.0, "WBC": 9.0,
    "Fibrinogen": 250.0, "Platelets": 220.0,
}

POPULATION_STDS = {
    "HR": 18.0, "O2Sat": 3.2, "Temp": 0.8, "SBP": 22.0,
    "MAP": 15.0, "DBP": 14.0, "Resp": 5.0, "EtCO2": 7.0,
    "BaseExcess": 4.5, "HCO3": 4.5, "FiO2": 0.25, "pH": 0.07,
    "PaCO2": 10.0, "SaO2": 5.0, "AST": 80.0, "BUN": 14.0,
    "Alkalinephos": 50.0, "Calcium": 1.2, "Chloride": 6.5,
    "Creatinine": 1.4, "Bilirubin_direct": 0.8, "Glucose": 45.0,
    "Lactate": 1.5, "Magnesium": 0.4, "Phosphate": 1.3,
    "Potassium": 0.65, "Bilirubin_total": 1.8, "TroponinI": 1.5,
    "Hct": 5.5, "Hgb": 2.1, "PTT": 16.0, "WBC": 7.0,
    "Fibrinogen": 110.0, "Platelets": 95.0,
}


# ============================================================
# Production Hospital Integration Settings
# ============================================================

class IntegrationConfig:
    """Security, auditing, and alert write-back integration settings."""
    # API security key (leave empty or set to None to disable authentication for debugging)
    api_key = os.getenv("SEPSIS_API_KEY", "sepsis_dev_secret_key_123")
    
    # Audit log path for logging all patient transactions (HIPAA requirement)
    audit_log_path = RESULTS_DIR / "audit.log"
    
    # EHR Alert Write-Back Webhook URL
    ehr_writeback_url = os.getenv("EHR_WRITEBACK_URL", "http://localhost:8000/api/mock-ehr/writeback")

