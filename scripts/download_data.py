"""
Data Download Helper Script.

Provides instructions and utilities for downloading the PhysioNet/CinC 2019
Challenge dataset for sepsis prediction.

The dataset requires PhysioNet credentialed access:
  1. Create an account at https://physionet.org
  2. Complete the required CITI training
  3. Request access to the dataset
  4. Download training_setA and training_setB

Usage:
    python scripts/download_data.py --help
    python scripts/download_data.py --check
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def check_data_availability() -> dict:
    """
    Check if the dataset is available in the expected directories.

    Returns:
        Dictionary with availability status.
    """
    result = {
        "hospital_a": False,
        "hospital_b": False,
        "hospital_a_count": 0,
        "hospital_b_count": 0,
    }

    if config.HOSPITAL_A_DIR.exists():
        psv_files = list(config.HOSPITAL_A_DIR.glob("*.psv"))
        result["hospital_a"] = len(psv_files) > 0
        result["hospital_a_count"] = len(psv_files)

    if config.HOSPITAL_B_DIR.exists():
        psv_files = list(config.HOSPITAL_B_DIR.glob("*.psv"))
        result["hospital_b"] = len(psv_files) > 0
        result["hospital_b_count"] = len(psv_files)

    return result


def print_instructions():
    """Print detailed download instructions."""
    print("=" * 62)
    print("    PhysioNet/CinC 2019 Challenge Dataset - Download Guide")
    print("=" * 62)
    print("""
This project uses the PhysioNet/Computing in Cardiology 2019
Challenge dataset for early sepsis prediction.

DATASET OVERVIEW:
  * ~40,000 ICU patients from 2 hospital systems
  * 40 clinical features per hourly timestep
  * 8 vital signs + 26 lab values + 6 demographics
  * Binary label: SepsisLabel (6-hour prediction window)

DOWNLOAD STEPS:

  1. Create a PhysioNet account:
     https://physionet.org/register/

  2. Complete the required CITI training:
     https://physionet.org/about/citi-course/

  3. Access the challenge dataset:
     https://physionet.org/content/challenge-2019/1.0.0/

  4. Download the training data:
     * training_setA.zip  (Hospital A, ~20,000 patients)
     * training_setB.zip  (Hospital B, ~20,000 patients)

  5. Extract the files to the following directories:
""")

    print(f"     Hospital A: {config.HOSPITAL_A_DIR}")
    print(f"     Hospital B: {config.HOSPITAL_B_DIR}")

    print("""
  The directory structure should look like:
     data/raw/training_setA/p000001.psv
     data/raw/training_setA/p000002.psv
     ...
     data/raw/training_setB/p100001.psv
     data/raw/training_setB/p100002.psv
     ...

ALTERNATIVE - Using wget (if credentialed):
  wget -r -N -c -np \\
    https://physionet.org/files/challenge-2019/1.0.0/ \\
    --user <your_username> --ask-password

==============================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset download helper for Sepsis Prediction System"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check if dataset is available"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show download instructions"
    )
    args = parser.parse_args()

    if args.check or not args.info:
        print("\nChecking dataset availability...")
        status = check_data_availability()

        print(f"\n  Hospital A directory: {config.HOSPITAL_A_DIR}")
        if status["hospital_a"]:
            print(f"    [OK] Found {status['hospital_a_count']} PSV files")
        else:
            print(f"    [MISSING] No PSV files found")

        print(f"\n  Hospital B directory: {config.HOSPITAL_B_DIR}")
        if status["hospital_b"]:
            print(f"    [OK] Found {status['hospital_b_count']} PSV files")
        else:
            print(f"    [MISSING] No PSV files found")

        if not status["hospital_a"] or not status["hospital_b"]:
            print("\n  [WARNING] Dataset not fully available. See instructions below.")
            print_instructions()
        else:
            print(f"\n  [SUCCESS] Dataset ready! Total: "
                  f"{status['hospital_a_count'] + status['hospital_b_count']} patients")

    if args.info:
        print_instructions()


if __name__ == "__main__":
    main()
