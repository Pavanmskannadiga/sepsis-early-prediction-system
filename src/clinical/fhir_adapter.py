"""
HL7 FHIR R4 Resource Generator for Sepsis Risk Assessment.

Exports standardized Fast Healthcare Interoperability Resources (FHIR R4)
compatible with modern Electronic Health Record (EHR) systems (Epic, Cerner, Allscripts).

Generated Resources:
- FHIR R4 Bundle (collection)
- Patient
- RiskAssessment (Sepsis AI Prediction & Stratification)
- Observation (Vital Signs & Labs)
- DiagnosticReport
"""

from typing import Dict, List, Optional
from datetime import datetime
import uuid


def generate_fhir_r4_bundle(
    patient_id: str,
    vitals: Dict[str, float],
    labs: Dict[str, float],
    risk_data: Dict,
    model_name: str = "Deep Sepsis Ensemble"
) -> Dict:
    """
    Constructs a valid FHIR R4 Bundle containing Patient, Observations,
    and RiskAssessment resources.
    """
    bundle_id = str(uuid.uuid4())
    now_iso = datetime.now().isoformat() + "Z"
    score = float(risk_data.get("risk_score", 0.0))
    risk_level = str(risk_data.get("risk_level", "LOW"))

    # Map LOINC codes for vitals/labs
    loinc_codes = {
        "HR": {"code": "8867-4", "display": "Heart rate", "unit": "beats/minute"},
        "O2Sat": {"code": "2708-6", "display": "Oxygen saturation in Arterial blood", "unit": "%"},
        "Temp": {"code": "8310-5", "display": "Body temperature", "unit": "Cel"},
        "SBP": {"code": "8480-6", "display": "Systolic blood pressure", "unit": "mm[Hg]"},
        "DBP": {"code": "8462-4", "display": "Diastolic blood pressure", "unit": "mm[Hg]"},
        "MAP": {"code": "8478-0", "display": "Mean blood pressure", "unit": "mm[Hg]"},
        "Resp": {"code": "9279-1", "display": "Respiratory rate", "unit": "breaths/minute"},
        "Lactate": {"code": "2276-4", "display": "Lactate in Blood", "unit": "mmol/L"},
        "WBC": {"code": "6690-2", "display": "Leukocytes in Blood", "unit": "10*3/uL"},
        "Creatinine": {"code": "2160-0", "display": "Creatinine in Serum or Plasma", "unit": "mg/dL"},
        "Platelets": {"code": "777-3", "display": "Platelets in Blood", "unit": "10*3/uL"},
    }

    entries = []

    # 1. Patient Resource
    patient_uuid = f"urn:uuid:{uuid.uuid4()}"
    entries.append({
        "fullUrl": patient_uuid,
        "resource": {
            "resourceType": "Patient",
            "id": patient_id,
            "identifier": [{
                "system": "urn:hospital:mrn",
                "value": patient_id
            }],
            "active": True
        }
    })

    # 2. Observation Resources for Vitals and Labs
    obs_references = []
    combined_telemetry = {**vitals, **labs}
    for feat, val in combined_telemetry.items():
        if feat in loinc_codes:
            obs_uuid = f"urn:uuid:{uuid.uuid4()}"
            meta = loinc_codes[feat]
            entries.append({
                "fullUrl": obs_uuid,
                "resource": {
                    "resourceType": "Observation",
                    "id": str(uuid.uuid4())[:8],
                    "status": "final",
                    "category": [{
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs" if feat in vitals else "laboratory",
                            "display": "Vital Signs" if feat in vitals else "Laboratory"
                        }]
                    }],
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": meta["code"],
                            "display": meta["display"]
                        }]
                    },
                    "subject": {"reference": patient_uuid},
                    "effectiveDateTime": now_iso,
                    "valueQuantity": {
                        "value": round(float(val), 2),
                        "unit": meta["unit"],
                        "system": "http://unitsofmeasure.org",
                        "code": meta["unit"]
                    }
                }
            })
            obs_references.append({"reference": obs_uuid})

    # 3. RiskAssessment Resource (Sepsis AI Prediction)
    risk_assessment_uuid = f"urn:uuid:{uuid.uuid4()}"
    entries.append({
        "fullUrl": risk_assessment_uuid,
        "resource": {
            "resourceType": "RiskAssessment",
            "id": f"sepsis-risk-{patient_id}",
            "status": "final",
            "subject": {"reference": patient_uuid},
            "occurrenceDateTime": now_iso,
            "basis": obs_references,
            "method": {
                "coding": [{
                    "system": "urn:sepsis-prediction:ai-model",
                    "code": model_name,
                    "display": f"{model_name} Deep Learning Sepsis Model"
                }]
            },
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "405623001",
                    "display": "Sepsis risk assessment (procedure)"
                }]
            },
            "prediction": [{
                "outcome": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": "91302008",
                        "display": "Sepsis (disorder)"
                    }]
                },
                "probabilityDecimal": score,
                "qualitativeRisk": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                        "code": risk_level.lower(),
                        "display": f"{risk_level} Risk of Sepsis"
                    }]
                },
                "whenPeriod": {
                    "start": now_iso
                }
            }],
            "note": [{
                "text": risk_data.get("interpretation", "AI Sepsis Prediction evaluated based on CinC 2019 PhysioNet criteria.")
            }]
        }
    })

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": now_iso,
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"],
            "lastUpdated": now_iso
        },
        "entry": entries
    }
