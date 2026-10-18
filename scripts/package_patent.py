"""
package_patent.py — Creates a zip file of the Sepsis Early Prediction System patent application.
"""
import os
import zipfile
from pathlib import Path

# Paths
ART_DIR = Path(r"C:\Users\PAVAN MS\.gemini\antigravity\brain\b888a668-b1da-429b-8945-89de43753019")
SPEC_MD = ART_DIR / "sepsis_patent_specification.md"
ZIP_OUT = ART_DIR / "sepsis_patent_package.zip"

def main():
    print(f"Reading specification from: {SPEC_MD}")
    if not SPEC_MD.exists():
        print(f"Error: {SPEC_MD} does not exist.")
        return

    content = SPEC_MD.read_text(encoding="utf-8")
    
    # Save a clean text version (removing some markdown symbols for clean print/submission)
    clean_txt = content.replace("### ", "").replace("## ", "").replace("# ", "").replace("**", "")
    spec_txt = ART_DIR / "sepsis_patent_specification.txt"
    spec_txt.write_text(clean_txt, encoding="utf-8")

    readme_content = """# Sepsis Early Prediction System — Patent Application Reference Bundle
This zip package contains the complete patent specification documents prepared for filing under the Patents Act, 1970 (Form 2 compliance).

## Contents:
1. `sepsis_patent_specification.md` — The complete specification formatted in clean markdown, containing drawings descriptions, details of the 8 clinical modules, mathematical formulations, and 10 defensible claims.
2. `sepsis_patent_specification.txt` — The raw plain text representation of the complete specification, ready for upload to official e-filing portals.

## Application Details:
* **Applicants**: PES Institute of Technology & Management, Pavana M. S., Megharaja S., Srujan S. Sagar, Abubakkar Siddiq, Prof. Shilpa K. C.
* **Technology**: Deep Learning (BiLSTM + Multi-Head Self-Attention), Class Imbalance Handling (Sequence-level SMOTE), Explainability (SHAP & attention heatmaps), and Real-time Gateways (HL7 V2 MLLP & FHIR Observation interface).

Prepared in 2026.
"""
    readme_path = ART_DIR / "README_patent.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    # Create Zip
    print(f"Creating Zip package at: {ZIP_OUT}")
    with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(SPEC_MD, arcname="sepsis_patent_specification.md")
        zipf.write(spec_txt, arcname="sepsis_patent_specification.txt")
        zipf.write(readme_path, arcname="README_patent.md")

    # Clean temporary files
    if spec_txt.exists():
        os.remove(spec_txt)
    if readme_path.exists():
        os.remove(readme_path)

    print("Success! Zip package created successfully.")

if __name__ == "__main__":
    main()
