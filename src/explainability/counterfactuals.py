"""
Clinical What-If Counterfactual Intervention Simulator.

Provides clinicians with real-time interactive simulation of therapeutic
interventions (fluids, vasopressors, oxygen, antibiotic response) to assess
how physiological stabilization affects deep learning sepsis onset risk.

Interventions modeled:
- IV Fluid Challenge: MAP increase, Lactate clearance
- Vasopressor Titration: MAP targeted restoration (65-80 mmHg)
- Supplemental Oxygen / BiPAP / Invasive Mech Vent: SpO2 elevation, RR normalization
- Antimicrobial / Source Control: WBC decline, Fever defervescence
"""

import copy
import numpy as np
from typing import Dict, List, Optional, Tuple


def simulate_intervention(
    current_vitals: Dict[str, float],
    current_labs: Dict[str, float],
    adjustments: Dict[str, float],
    predict_fn
) -> Dict:
    """
    Simulates the clinical impact of physiological parameter adjustments on sepsis risk.

    Args:
        current_vitals: Baseline vitals dictionary
        current_labs: Baseline labs dictionary
        adjustments: Dict of feature deltas or target values (e.g. {'MAP': 75, 'Lactate': 1.8})
        predict_fn: Function (vitals, labs) -> (risk_score, details)

    Returns:
        Dict comparing baseline vs counterfactual risk and delta breakdown.
    """
    # 1. Evaluate baseline
    base_score, base_contribs = predict_fn(current_vitals, current_labs)

    # 2. Apply physiological adjustments
    sim_vitals = copy.deepcopy(current_vitals)
    sim_labs = copy.deepcopy(current_labs)

    applied_changes = []
    for param, new_val in adjustments.items():
        if param in sim_vitals:
            old_val = sim_vitals[param]
            sim_vitals[param] = float(new_val)
            applied_changes.append({
                "parameter": param,
                "baseline": round(old_val, 2),
                "simulated": round(float(new_val), 2),
                "delta": round(float(new_val) - old_val, 2)
            })
        elif param in sim_labs:
            old_val = sim_labs[param]
            sim_labs[param] = float(new_val)
            applied_changes.append({
                "parameter": param,
                "baseline": round(old_val, 2),
                "simulated": round(float(new_val), 2),
                "delta": round(float(new_val) - old_val, 2)
            })

    # 3. Re-evaluate with simulated state
    sim_score, sim_contribs = predict_fn(sim_vitals, sim_labs)

    delta_risk = float(sim_score - base_score)
    pct_reduction = float((base_score - sim_score) / max(base_score, 1e-4) * 100.0) if delta_risk < 0 else 0.0

    return {
        "baseline_risk": round(float(base_score), 4),
        "simulated_risk": round(float(sim_score), 4),
        "delta_risk": round(delta_risk, 4),
        "risk_reduction_pct": round(pct_reduction, 1),
        "applied_changes": applied_changes,
        "improved": delta_risk < -0.01,
        "clinical_summary": (
            f"Simulated intervention reduced risk from {base_score*100:.1f}% to {sim_score*100:.1f}% "
            f"({pct_reduction:.1f}% relative reduction)."
            if delta_risk < 0 else
            f"Simulated parameters result in risk change from {base_score*100:.1f}% to {sim_score*100:.1f}%."
        )
    }
