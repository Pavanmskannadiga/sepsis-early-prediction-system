"""
Surviving Sepsis Campaign (SSC) Hour-1 Protocol Generator.

Translates real-time patient physiological observations, laboratory values,
and AI risk predictions into guideline-adherent clinical bundle recommendations
based on the international Surviving Sepsis Campaign Guidelines.

The Hour-1 Sepsis Bundle includes:
1. Measure blood lactate level (re-measure if initial > 2 mmol/L).
2. Obtain blood cultures prior to administration of antibiotics.
3. Administer broad-spectrum antimicrobials.
4. Rapidly administer 30 mL/kg crystalloid for hypotension (MAP < 65 mmHg) or lactate >= 4.0 mmol/L.
5. Apply vasopressors if hypotensive during or after fluid resuscitation to maintain MAP >= 65 mmHg.
"""

from typing import Dict, List, Optional
from datetime import datetime


def generate_sepsis_bundle(
    vitals: Dict[str, float],
    labs: Dict[str, float],
    risk_score: float,
    patient_weight_kg: float = 70.0
) -> Dict:
    """
    Evaluates patient telemetry and generates a Surviving Sepsis Campaign Hour-1 Action Bundle.

    Args:
        vitals: Latest vital signs dict (HR, MAP, SBP, Resp, Temp, O2Sat)
        labs: Latest lab values dict (Lactate, WBC, Creatinine, Platelets, pH)
        risk_score: Sepsis prediction score [0, 1]
        patient_weight_kg: Estimated or recorded patient weight in kg

    Returns:
        Dict with clinical bundle items, urgency level, and physiological triggers.
    """
    map_val = vitals.get("MAP", 80.0)
    sbp_val = vitals.get("SBP", 120.0)
    lactate_val = labs.get("Lactate", 1.2)
    resp_val = vitals.get("Resp", 16.0)
    hr_val = vitals.get("HR", 75.0)
    temp_val = vitals.get("Temp", 37.0)

    is_alert = risk_score >= 0.55
    is_critical = risk_score >= 0.78
    is_hypotensive = map_val < 65.0 or sbp_val < 90.0
    is_severe_lactate = lactate_val >= 4.0
    is_elevated_lactate = lactate_val > 2.0

    actions: List[Dict] = []

    # 1. Lactate Measurement
    if is_severe_lactate:
        actions.append({
            "id": "ACT_LACTATE_CRITICAL",
            "priority": "IMMEDIATE",
            "category": "Diagnostics",
            "action": f"Critical Lactate ({lactate_val:.1f} mmol/L >= 4.0). Initiate immediate resuscitation and serial lactate tracking q2-4h.",
            "guideline": "SSC Hour-1 Bundle",
            "rationale": "Severe tissue hypoperfusion and anaerobic metabolism."
        })
    elif is_elevated_lactate:
        actions.append({
            "id": "ACT_LACTATE_ELEVATED",
            "priority": "HIGH",
            "category": "Diagnostics",
            "action": f"Elevated Lactate ({lactate_val:.1f} mmol/L > 2.0). Re-measure within 2-4 hours to guide resuscitation.",
            "guideline": "SSC Hour-1 Bundle",
            "rationale": "Monitor clearance to assess restoration of tissue perfusion."
        })
    elif is_alert:
        actions.append({
            "id": "ACT_LACTATE_BASELINE",
            "priority": "STANDARD",
            "category": "Diagnostics",
            "action": "Obtain baseline serum lactate level.",
            "guideline": "SSC Hour-1 Bundle",
            "rationale": "Identify occult metabolic distress."
        })

    # 2. Blood Cultures
    if is_alert:
        actions.append({
            "id": "ACT_BLOOD_CULTURES",
            "priority": "HIGH",
            "category": "Microbiology",
            "action": "Obtain 2 sets of blood cultures (aerobic and anaerobic) BEFORE initiating new antimicrobial therapy.",
            "guideline": "SSC Hour-1 Bundle",
            "rationale": "Optimize pathogen identification without delaying antibiotic administration (>45 min)."
        })

    # 3. Empiric Antimicrobials
    if is_alert:
        actions.append({
            "id": "ACT_ANTIBIOTICS",
            "priority": "IMMEDIATE" if is_critical else "HIGH",
            "category": "Therapeutics",
            "action": "Administer IV broad-spectrum antimicrobial therapy targeting suspected infectious source.",
            "guideline": "SSC Hour-1 Bundle",
            "rationale": "Each hour of antimicrobial delay increases septic shock mortality by ~7.6%."
        })

    # 4. Fluid Resuscitation
    fluid_dose_ml = round(patient_weight_kg * 30.0)
    if is_hypotensive or is_severe_lactate:
        actions.append({
            "id": "ACT_FLUID_RESUSCITATION",
            "priority": "IMMEDIATE",
            "category": "Resuscitation",
            "action": f"Administer 30 mL/kg IV crystalloid fluid bolus ({fluid_dose_ml} mL total, e.g., Plasmalyte / Ringer's Lactate) within first 3 hours.",
            "guideline": "SSC Hour-1 Bundle",
            "rationale": "Restore effective circulating intravascular volume and cardiac preload."
        })

    # 5. Vasopressor Support
    if is_hypotensive:
        actions.append({
            "id": "ACT_VASOPRESSORS",
            "priority": "IMMEDIATE",
            "category": "Hemodynamics",
            "action": "Initiate IV Norepinephrine infusion (1st line). Titrate rapidly to achieve and maintain MAP >= 65 mmHg.",
            "guideline": "SSC Hour-1 Bundle",
            "rationale": "Persistent hypotension despite fluid challenge indicates vasoplegic septic shock."
        })

    # 6. Respiratory and Oxygen Support
    if vitals.get("O2Sat", 98.0) < 92.0 or resp_val > 24.0:
        actions.append({
            "id": "ACT_OXYGEN",
            "priority": "HIGH",
            "category": "Respiratory",
            "action": f"Titrate supplemental O2 to maintain SpO2 >= 94%. Current SpO2={vitals.get('O2Sat', 98.0):.0f}%, RR={resp_val:.0f}/min.",
            "guideline": "Supportive Care",
            "rationale": "Prevent hypoxic end-organ damage and reduce work of breathing."
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "risk_level": "CRITICAL" if is_critical else ("HIGH" if is_alert else "MONITORING"),
        "risk_score": round(risk_score, 4),
        "protocol_name": "Surviving Sepsis Campaign (SSC) 1-Hour Bundle",
        "weight_used_kg": patient_weight_kg,
        "actions_required_count": len(actions),
        "actions": actions
    }
