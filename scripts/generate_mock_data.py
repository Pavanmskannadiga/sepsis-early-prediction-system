"""
Mock Data Generator.

Generates simulated patient files (.psv) for both Hospital A and Hospital B
to enable immediate local execution of the training, evaluation, and explanation
pipelines without requiring the full PhysioNet dataset.

Creates:
  - 50 patient records in data/raw/training_setA
  - 50 patient records in data/raw/training_setB
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def generate_mock_patients(directory: Path, num_patients: int = 50, seed: int = 42):
    directory.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    cols = config.ALL_FEATURES + [config.TARGET_COLUMN]

    print(f"Generating {num_patients} mock patient files in: {directory}")

    for i in range(num_patients):
        num_hours = np.random.randint(12, 72)
        data = {}

        # Set demographics
        age = np.random.randint(18, 90)
        gender = np.random.randint(0, 2)
        unit1 = np.random.randint(0, 2)
        unit2 = 1 - unit1
        hosp_adm_time = -np.random.uniform(0.1, 24.0)

        # Determine if this patient will develop sepsis
        has_sepsis = np.random.rand() > 0.75
        sepsis_onset_hour = np.random.randint(num_hours // 2, num_hours) if has_sepsis else -1

        for col in config.ALL_FEATURES:
            if col == "Age":
                data[col] = [age] * num_hours
            elif col == "Gender":
                data[col] = [gender] * num_hours
            elif col == "Unit1":
                data[col] = [unit1] * num_hours
            elif col == "Unit2":
                data[col] = [unit2] * num_hours
            elif col == "HospAdmTime":
                data[col] = [hosp_adm_time] * num_hours
            elif col == "ICULOS":
                data[col] = list(range(1, num_hours + 1))
            else:
                # Get normal/median baseline value
                median_val = config.POPULATION_MEDIANS.get(col, 0.0)
                
                # Sepsis patients have abnormal trajectories near onset
                if has_sepsis and col in ["HR", "Resp", "Temp", "Lactate", "WBC"]:
                    # Create an escalating trend
                    base_vals = np.random.randn(num_hours) * (median_val * 0.05) + median_val
                    # Trend starts escalating 12 hours before onset
                    start_trend = max(0, sepsis_onset_hour - 12)
                    trend_len = num_hours - start_trend
                    
                    if col in ["HR", "Resp", "Lactate"]:
                        # Escalate values upward
                        escalation = np.linspace(0, median_val * 0.4, trend_len)
                        base_vals[start_trend:] += escalation
                    elif col == "Temp":
                        # Fever spike
                        escalation = np.linspace(0, 2.0, trend_len)
                        base_vals[start_trend:] += escalation
                    elif col == "WBC":
                        escalation = np.linspace(0, median_val * 0.5, trend_len)
                        base_vals[start_trend:] += escalation
                
                elif has_sepsis and col in ["SBP", "MAP", "DBP", "Platelets"]:
                    # Create a declining trend (shock indicators)
                    base_vals = np.random.randn(num_hours) * (median_val * 0.05) + median_val
                    start_trend = max(0, sepsis_onset_hour - 12)
                    trend_len = num_hours - start_trend
                    
                    # Decline values downward
                    decline = np.linspace(0, median_val * 0.25, trend_len)
                    base_vals[start_trend:] -= decline
                
                else:
                    # Healthy random walk around clinical median
                    base_vals = np.random.randn(num_hours) * (median_val * 0.05) + median_val

                # Introduce sparse missing values in labs (approx 80% missingness)
                if col in config.LAB_VALUES:
                    missing_mask = np.random.rand(num_hours) > 0.85
                    # Ensure at least one measurement exists to avoid backward fill warnings
                    missing_mask[0] = True 
                    base_vals[~missing_mask] = np.nan

                data[col] = base_vals

        # SepsisLabel starts 6 hours before actual onset
        sepsis_label = np.zeros(num_hours, dtype=int)
        if has_sepsis:
            sepsis_label[max(0, sepsis_onset_hour - 6):] = 1
        data[config.TARGET_COLUMN] = sepsis_label

        df = pd.DataFrame(data, columns=cols)
        df.to_csv(directory / f"p_{i:06d}.psv", sep="|", index=False)


def main():
    print("Initializing simulated clinical dataset for Sepsis Prediction...")
    generate_mock_patients(config.HOSPITAL_A_DIR, num_patients=50, seed=42)
    generate_mock_patients(config.HOSPITAL_B_DIR, num_patients=50, seed=43)
    print("\n[SUCCESS] Mock patient files generated successfully.")
    print("You can now run training, evaluation, and explanation scripts natively.")


if __name__ == "__main__":
    main()
