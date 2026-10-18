"""
ICU Data Stream Simulator
==========================
Generates realistic, continuous patient vital sign and lab streams that
mimic actual ICU bedside monitor output.

Each simulated patient has:
  - A clinical trajectory: stable | deteriorating | septic | recovering
  - Physiological noise added to every reading
  - Lab values that update every 4-6 hours (sparse, as in real ICUs)
  - Correlated vital signs (e.g., HR rises when O2Sat falls)

This module is used when no real data feed is connected.
In production, replace `SimulatedPatient.next_reading()` with a call to
your HL7 / FHIR / CSV file-watch adapter.
"""

import sys
import math
import random
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ──────────────────────────────────────────────────────────────────────────────
# Clinical normal ranges (used as trajectory anchors)
# ──────────────────────────────────────────────────────────────────────────────
NORMAL = {
    "HR": 75, "O2Sat": 97, "Temp": 36.9, "SBP": 118, "MAP": 82,
    "DBP": 65, "Resp": 16, "EtCO2": 38,
    "Lactate": 1.2, "Creatinine": 1.0, "WBC": 8.5, "pH": 7.40,
    "Platelets": 220, "Bilirubin_total": 0.6, "HCO3": 24,
    "FiO2": 0.21, "PaCO2": 40, "BUN": 15, "Hct": 38, "Glucose": 110,
}

SEPSIS_TARGET = {
    "HR": 128, "O2Sat": 89, "Temp": 39.1, "SBP": 88, "MAP": 58,
    "DBP": 48, "Resp": 30, "EtCO2": 28,
    "Lactate": 5.8, "Creatinine": 3.2, "WBC": 19.5, "pH": 7.22,
    "Platelets": 85, "Bilirubin_total": 2.8, "HCO3": 16,
    "FiO2": 0.55, "PaCO2": 48, "BUN": 38, "Hct": 30, "Glucose": 165,
}

NOISE_STD = {
    "HR": 3, "O2Sat": 0.8, "Temp": 0.08, "SBP": 5, "MAP": 4,
    "DBP": 3, "Resp": 1.2, "EtCO2": 1.5,
    "Lactate": 0.12, "Creatinine": 0.05, "WBC": 0.4, "pH": 0.005,
    "Platelets": 5, "Bilirubin_total": 0.04, "HCO3": 0.5,
    "FiO2": 0.01, "PaCO2": 1.5, "BUN": 1.0, "Hct": 0.8, "Glucose": 6,
}

# ──────────────────────────────────────────────────────────────────────────────
# Patient Trajectories
# ──────────────────────────────────────────────────────────────────────────────
TRAJECTORIES = {
    "stable":       {"label": 0, "drift_rate": 0.000, "sepsis_prob": 0.02},
    "deteriorating":{"label": 0, "drift_rate": 0.004, "sepsis_prob": 0.12},
    "septic":       {"label": 1, "drift_rate": 0.010, "sepsis_prob": 0.85},
    "recovering":   {"label": 0, "drift_rate": -0.003,"sepsis_prob": 0.05},
}

WARD_NAMES = [f"ICU-{i:02d}" for i in range(1, 25)]

# ──────────────────────────────────────────────────────────────────────────────
# SimulatedPatient
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SimulatedPatient:
    patient_id: str
    name: str
    age: int
    gender: str
    bed: str
    trajectory: str
    admission_time: datetime = field(default_factory=datetime.now)

    # Internal state
    current_values: Dict[str, float] = field(default_factory=dict)
    icu_hours: float = 0.0
    sepsis_onset_hour: Optional[float] = None
    history: List[Dict] = field(default_factory=list)
    lab_last_updated: float = -99.0  # hours since admission

    def __post_init__(self):
        # Start from normal values with slight personal variation
        self.current_values = {
            k: v + random.gauss(0, NOISE_STD.get(k, 0.1) * 2)
            for k, v in NORMAL.items()
        }
        self.current_values["Age"] = float(self.age)
        self.current_values["Gender"] = 1.0 if self.gender == "M" else 0.0
        self.current_values["ICULOS"] = 0.0
        self.current_values["Unit1"] = random.choice([0.0, 1.0])
        self.current_values["Unit2"] = 1.0 - self.current_values["Unit1"]
        self.current_values["HospAdmTime"] = random.uniform(-6, -1)
        self.current_values["SepsisLabel"] = 0.0

        # Septic patients are already partially deteriorated
        if self.trajectory == "septic":
            t = random.uniform(0.3, 0.6)
            for k in NORMAL:
                if k in SEPSIS_TARGET:
                    self.current_values[k] = NORMAL[k] + t * (SEPSIS_TARGET[k] - NORMAL[k])

    def _drift_toward(self, key: str, t: float) -> float:
        """Physiologically bounded drift with Ornstein-Uhlenbeck mean reversion."""
        current = self.current_values.get(key, NORMAL.get(key, 0.0))
        noise = random.gauss(0, NOISE_STD.get(key, 0.1))

        if self.trajectory == "stable":
            target = NORMAL.get(key, current)
            # Gentle pull toward normal, preventing unbounded random walk
            pull = 0.20 * (target - current)
            new_val = current + pull + noise * 0.7
        elif self.trajectory == "recovering":
            target = NORMAL.get(key, current)
            # Smooth recovery back to healthy ranges
            pull = 0.15 * (target - current)
            new_val = current + pull + noise * 0.8
        elif self.trajectory == "septic":
            target = SEPSIS_TARGET.get(key, current)
            # Fluctuates around sepsis target with clinical volatility
            pull = 0.15 * (target - current)
            new_val = current + pull + noise
        elif self.trajectory == "deteriorating":
            target = SEPSIS_TARGET.get(key, current)
            # Gradual deterioration toward sepsis
            pull = 0.05 * (target - current)
            new_val = current + pull + noise
        else:
            target = NORMAL.get(key, current)
            pull = 0.15 * (target - current)
            new_val = current + pull + noise

        return new_val

    def _correlate_vitals(self):
        """Apply physiological correlations between vital signs."""
        cv = self.current_values
        # O2Sat drops when Resp is very high (respiratory fatigue)
        if cv.get("Resp", 16) > 28:
            cv["O2Sat"] = min(cv.get("O2Sat", 97), 93.0)
        # HR rises when MAP drops (compensatory tachycardia, bounded realistically)
        if cv.get("MAP", 82) < 65:
            cv["HR"] = min(145.0, max(cv.get("HR", 75), 108.0))
        # pH drops with lactate rise (metabolic acidosis)
        lactate = cv.get("Lactate", 1.2)
        if lactate > 3:
            cv["pH"] = min(cv.get("pH", 7.4), 7.4 - (lactate - 3) * 0.04)
        # HCO3 drops with acidosis
        cv["HCO3"] = max(10, 24 - max(0, 7.4 - cv.get("pH", 7.4)) * 150)

    def _clamp_all(self):
        """Clamp all values to physiologically plausible ranges."""
        clamps = {
            "HR": (40, 190), "O2Sat": (70, 100), "Temp": (35.0, 41.5),
            "SBP": (60, 220), "MAP": (40, 140), "DBP": (30, 110),
            "Resp": (8, 45), "EtCO2": (15, 60),
            "Lactate": (0.4, 15.0), "Creatinine": (0.4, 10.0),
            "WBC": (1.5, 35.0), "pH": (7.0, 7.6),
            "Platelets": (15, 450), "Bilirubin_total": (0.2, 20),
            "HCO3": (8, 40), "FiO2": (0.21, 1.0),
            "PaCO2": (20, 80), "BUN": (5, 120), "Hct": (15, 55),
            "Glucose": (50, 450),
        }
        for k, (lo, hi) in clamps.items():
            if k in self.current_values:
                self.current_values[k] = max(lo, min(hi, self.current_values[k]))

    def next_reading(self) -> Dict:
        """Advance simulation by 1 hour and return a new reading dict."""
        self.icu_hours += 1.0
        # Realistic stay cycle: in an ICU ward, stays cycle between 4 and 48 hours
        if self.icu_hours > 72.0:
            self.icu_hours = round(random.uniform(6.0, 18.0), 1)
            self.history = self.history[-18:]
        self.current_values["ICULOS"] = self.icu_hours

        # Update vitals every tick
        vital_keys = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
        for k in vital_keys:
            self.current_values[k] = self._drift_toward(k, self.icu_hours)

        # Update labs every ~4–6 hours
        if self.icu_hours - self.lab_last_updated >= random.uniform(4, 6):
            lab_keys = ["Lactate", "Creatinine", "WBC", "pH", "Platelets",
                        "Bilirubin_total", "HCO3", "FiO2", "PaCO2", "BUN",
                        "Hct", "Glucose"]
            for k in lab_keys:
                self.current_values[k] = self._drift_toward(k, self.icu_hours)
            self.lab_last_updated = self.icu_hours

        self._correlate_vitals()
        self._clamp_all()

        # Sepsis onset determination
        traj = TRAJECTORIES[self.trajectory]
        if self.sepsis_onset_hour is None:
            cumulative_prob = min(0.98, traj["sepsis_prob"] * (1 + self.icu_hours * 0.02))
            if random.random() < cumulative_prob * 0.01:  # per-tick probability
                self.sepsis_onset_hour = self.icu_hours
        self.current_values["SepsisLabel"] = 1.0 if (
            self.sepsis_onset_hour and self.icu_hours >= self.sepsis_onset_hour
        ) else 0.0

        reading = dict(self.current_values)
        reading["timestamp"] = datetime.now().isoformat()
        reading["icu_hours"] = self.icu_hours
        reading["trajectory"] = self.trajectory
        self.history.append(reading)
        # Keep last 48 readings (window)
        if len(self.history) > 48:
            self.history = self.history[-48:]
        return reading

    def get_feature_window(self) -> Optional[np.ndarray]:
        """Return a (T, 34) numpy array of CLINICAL_FEATURES for the last T hours."""
        if len(self.history) < 4:
            return None
        feats = config.CLINICAL_FEATURES  # 34 features
        rows = []
        for h in self.history[-48:]:
            row = [h.get(f, config.POPULATION_MEDIANS.get(f, 0.0)) for f in feats]
            rows.append(row)
        return np.array(rows, dtype=np.float32)  # (T, 34)

    def to_summary(self) -> Dict:
        """Compact dict for WebSocket broadcast."""
        cv = self.current_values
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "bed": self.bed,
            "trajectory": self.trajectory,
            "icu_hours": round(self.icu_hours, 1),
            "sepsis_onset_hour": self.sepsis_onset_hour,
            "vitals": {
                "HR":    round(cv.get("HR", 75), 1),
                "O2Sat": round(cv.get("O2Sat", 97), 1),
                "Temp":  round(cv.get("Temp", 36.9), 1),
                "SBP":   round(cv.get("SBP", 118), 0),
                "MAP":   round(cv.get("MAP", 82), 0),
                "DBP":   round(cv.get("DBP", 65), 0),
                "Resp":  round(cv.get("Resp", 16), 1),
            },
            "labs": {
                "Lactate":    round(cv.get("Lactate", 1.2), 2),
                "Creatinine": round(cv.get("Creatinine", 1.0), 2),
                "WBC":        round(cv.get("WBC", 8.5), 1),
                "pH":         round(cv.get("pH", 7.40), 3),
                "Platelets":  round(cv.get("Platelets", 220), 0),
            },
            "history_length": len(self.history),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Ward — manages a set of patients
# ──────────────────────────────────────────────────────────────────────────────
PATIENT_NAMES = [
    ("Arjun", "Sharma"), ("Priya", "Nair"), ("Ravi", "Kumar"), ("Sunita", "Patel"),
    ("Mohammed", "Khan"), ("Lakshmi", "Rao"), ("Suresh", "Verma"), ("Anjali", "Gupta"),
    ("David", "Chen"), ("Elena", "Rostova"), ("Kavita", "Deshmukh"), ("Marcus", "Brody"),
    ("Fatima", "Zahra"), ("Rajesh", "Pillai"), ("Grace", "O'Connor"), ("Vikram", "Malhotra"),
    ("Aisha", "Al-Mansoor"), ("Lucas", "Silva"), ("Deepa", "Iyer"), ("Samuel", "Okoye"),
    ("Meera", "Joshi"), ("Hannah", "Schmidt"), ("Karthik", "Rangan"), ("Chloe", "Dubois"),
]

class ICUWard:
    """Manages all simulated patients in the ICU."""

    def __init__(self, n_patients: int = 12):
        self.patients: Dict[str, SimulatedPatient] = {}
        self._create_patients(n_patients)

    def _create_patients(self, n: int):
        trajectories = [
            "stable", "deteriorating", "septic", "recovering",
            "stable", "stable", "deteriorating", "septic",
            "recovering", "stable", "deteriorating", "stable",
            "stable", "septic", "recovering", "deteriorating"
        ]
        for i in range(n):
            fname, lname = PATIENT_NAMES[i % len(PATIENT_NAMES)]
            pid = f"P{1000 + i + 1}"
            traj = trajectories[i % len(trajectories)]
            p = SimulatedPatient(
                patient_id=pid,
                name=f"{fname} {lname}",
                age=random.randint(42, 84),
                gender=random.choice(["M", "F"]),
                bed=WARD_NAMES[i % len(WARD_NAMES)],
                trajectory=traj,
            )
            # Pre-warm simulation so patients have some history
            pre_hours = random.randint(4, 24)
            for _ in range(pre_hours):
                p.next_reading()
            self.patients[pid] = p

    def tick_all(self) -> List[Dict]:
        """Advance all patients by 1 hour and return their summaries."""
        summaries = []
        for p in self.patients.values():
            p.next_reading()
            summaries.append(p.to_summary())
        return summaries

    def get_patient(self, pid: str) -> Optional[SimulatedPatient]:
        return self.patients.get(pid)

    def all_summaries(self) -> List[Dict]:
        return [p.to_summary() for p in self.patients.values()]
