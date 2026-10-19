"""
Live Sepsis Prediction — FastAPI Backend Server
================================================

Endpoints:
  GET  /                       → Serve dashboard (index.html)
  GET  /api/status             → Server + model status
  GET  /api/patients           → All current patient summaries
  GET  /api/patients/{pid}     → Single patient detail
  POST /api/ingest             → Push real machine data (JSON body)
  WS   /ws/monitor             → Live broadcast (5-second ticks)
  WS   /ws/patient/{pid}       → Per-patient stream

Usage:
  python backend/server.py
  OR
  python start_server.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import asyncio
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Security, Depends, status as fastapi_status
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import httpx



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from backend.icu_simulator import ICUWard, SimulatedPatient
from backend.inference_engine import InferenceEngine

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sepsis Prediction System — Live API",
    description="Real-time sepsis risk monitoring with BiLSTM+Attention model",
    version="2.0.0",
)

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"

# ──────────────────────────────────────────────────────────────────────────────
# Global state
import os
ICU_PATIENT_COUNT = int(os.getenv("ICU_PATIENT_COUNT", "12"))
ward = ICUWard(n_patients=ICU_PATIENT_COUNT)
engine = InferenceEngine()

# Latest predictions cache  {patient_id: prediction_dict}
_latest_predictions: Dict[str, Dict] = {}

# WebSocket connection pools
_ward_connections: Set[WebSocket] = set()
_patient_connections: Dict[str, Set[WebSocket]] = {}

# Tick counter
_tick_count = 0
_server_start = datetime.now()

# ──────────────────────────────────────────────────────────────────────────────
# Production Hospital Integration Helpers
# ──────────────────────────────────────────────────────────────────────────────
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)):
    """Verifies the incoming X-API-Key header matches the configured key."""
    expected_key = getattr(config.IntegrationConfig, "api_key", None)
    if not expected_key:
        return None  # Auth disabled
    if api_key != expected_key:
        raise HTTPException(
            status_code=fastapi_status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials. Invalid X-API-Key."
        )
    return api_key

def write_audit_log(patient_id: str, action: str, details: dict):
    """Appends an event to the HIPAA audit log."""
    log_path = getattr(config.IntegrationConfig, "audit_log_path", None)
    if not log_path:
        return
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "action": action,
        "details": details
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[Audit Log Error] Failed to write audit log: {e}")

async def dispatch_ehr_alert(patient_id: str, score: float, risk_level: str, explanation: dict):
    """Dispatches a sepsis alert to the EHR writeback endpoint (configured URL)."""
    writeback_url = getattr(config.IntegrationConfig, "ehr_writeback_url", None)
    if not writeback_url:
        return
    
    payload = {
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "sepsis_probability": score,
        "risk_level": risk_level,
        "alert_triggered": risk_level in ["HIGH", "CRITICAL"],
        "clinical_evidence": explanation
    }
    
    headers = {}
    expected_key = getattr(config.IntegrationConfig, "api_key", None)
    if expected_key:
        headers["X-API-Key"] = expected_key
        
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(writeback_url, json=payload, headers=headers)
            if resp.status_code == 200:
                print(f"[EHR Write-Back] Dispatched alert for {patient_id} successfully.")
            else:
                print(f"[EHR Write-Back Error] Endpoint returned {resp.status_code} for {patient_id}.")
    except Exception as e:
        # Ignore connection errors during local simulation to avoid cluttering console
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Background tick loop  (runs in asyncio event loop)
# ──────────────────────────────────────────────────────────────────────────────
TICK_INTERVAL_SECONDS = 5   # Every 5 seconds = 1 simulated hour of data

async def _tick_loop():
    """Advance all patients and broadcast predictions to all WebSocket clients."""
    global _tick_count, _ward_connections, _patient_connections
    while True:
        await asyncio.sleep(TICK_INTERVAL_SECONDS)
        _tick_count += 1

        # Advance simulation
        summaries = ward.tick_all()

        # Run inference on each patient
        broadcast_payload = []
        alerts = []
        for summary in summaries:
            pid = summary["patient_id"]
            patient = ward.get_patient(pid)
            window = patient.get_feature_window() if patient else None

            prediction = engine.predict(summary, feature_window=window)
            
            # Audit log entry for tick prediction (essential for clinical operations auditing)
            write_audit_log(pid, "prediction", {
                "score": prediction.get("risk_score"),
                "level": prediction.get("risk_level"),
                "alert": prediction.get("alert")
            })
            
            # EHR Alert Writeback integration (High / Critical alerts sent to local connector)
            if prediction.get("alert") or prediction.get("risk_level") in ["HIGH", "CRITICAL"]:
                await dispatch_ehr_alert(pid, prediction["risk_score"], prediction["risk_level"], {
                    "top_features": prediction.get("top_features"),
                    "clinical_scores": prediction.get("clinical_scores")
                })

            # Merge prediction into summary
            payload = {**summary, **prediction, "tick": _tick_count}
            _latest_predictions[pid] = payload
            broadcast_payload.append(payload)
            if prediction.get("alert"):
                alerts.append({"patient_id": pid, "name": summary["name"],
                               "bed": summary["bed"], "risk_score": prediction["risk_score"],
                               "risk_level": prediction["risk_level"]})

        # Build ward broadcast
        ward_msg = json.dumps({
            "type": "ward_update",
            "tick": _tick_count,
            "timestamp": datetime.now().isoformat(),
            "patients": broadcast_payload,
            "alerts": alerts,
            "model": engine._get_model_display_name() if engine.model_loaded else "Heuristic",
        })

        # Broadcast to ward monitors
        dead = set()
        for ws in list(_ward_connections):
            try:
                await ws.send_text(ward_msg)
            except Exception:
                dead.add(ws)
        _ward_connections -= dead

        # Broadcast to per-patient monitors
        for pid, pconns in list(_patient_connections.items()):
            pdata = _latest_predictions.get(pid)
            if not pdata:
                continue
            pmsg = json.dumps({"type": "patient_update", **pdata})
            dead_p = set()
            for ws in list(pconns):
                try:
                    await ws.send_text(pmsg)
                except Exception:
                    dead_p.add(ws)
            pconns -= dead_p


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_tick_loop())
    print(f"\n{'='*60}")
    print(f"  [LIVE] Sepsis Prediction Server")
    print(f"  Dashboard : http://localhost:8000/live.html")
    print(f"  API docs  : http://localhost:8000/docs")
    print(f"  Model     : {engine._get_model_display_name() + ' [OK]' if engine.model_loaded else 'Clinical Heuristic [fallback]'}")
    print(f"  Patients  : {len(ward.patients)} simulated ICU patients")
    print(f"  Tick rate : every {TICK_INTERVAL_SECONDS}s")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(DASHBOARD_DIR / "index.html"))

@app.get("/index.html", include_in_schema=False)
async def index_html():
    return FileResponse(str(DASHBOARD_DIR / "index.html"))

@app.get("/live.html", include_in_schema=False)
async def live_dashboard():
    return FileResponse(str(DASHBOARD_DIR / "live.html"))

@app.get("/offline_engine.js", include_in_schema=False)
async def offline_engine_js():
    return FileResponse(str(DASHBOARD_DIR / "offline_engine.js"), media_type="application/javascript")

@app.get("/dashboard.js", include_in_schema=False)
async def dashboard_js():
    return FileResponse(str(DASHBOARD_DIR / "dashboard.js"), media_type="application/javascript")

@app.get("/style.css", include_in_schema=False)
async def style_css():
    return FileResponse(str(DASHBOARD_DIR / "style.css"), media_type="text/css")

@app.get("/manifest.json", include_in_schema=False)
async def manifest_json():
    return FileResponse(str(DASHBOARD_DIR / "manifest.json"), media_type="application/manifest+json")

@app.get("/sw.js", include_in_schema=False)
async def service_worker_js():
    return FileResponse(str(DASHBOARD_DIR / "sw.js"), media_type="application/javascript")

@app.get("/api/status")
async def status():
    return {
        "status": "running",
        "server_start": _server_start.isoformat(),
        "uptime_seconds": (datetime.now() - _server_start).seconds,
        "tick_count": _tick_count,
        "model_loaded": engine.model_loaded,
        "model_name": engine._get_model_display_name() if engine.model_loaded else "Clinical Heuristic",
        "n_patients": len(ward.patients),
        "ws_connections": len(_ward_connections),
        "tick_interval_s": TICK_INTERVAL_SECONDS,
    }

@app.get("/api/patients")
async def get_all_patients():
    summaries = ward.all_summaries()
    results = []
    for s in summaries:
        pred = _latest_predictions.get(s["patient_id"], {})
        results.append({**s, **pred})
    return {"patients": results, "count": len(results)}

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str):
    patient = ward.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    summary = patient.to_summary()
    pred = _latest_predictions.get(patient_id, {})
    history = patient.history[-48:]
    return {**summary, **pred, "history": history}


# ── Real machine data ingestion ─────────────────────────────────────────────
class MachineReading(BaseModel):
    patient_id: str
    HR: Optional[float] = None
    O2Sat: Optional[float] = None
    Temp: Optional[float] = None
    SBP: Optional[float] = None
    MAP: Optional[float] = None
    DBP: Optional[float] = None
    Resp: Optional[float] = None
    Lactate: Optional[float] = None
    Creatinine: Optional[float] = None
    WBC: Optional[float] = None
    pH: Optional[float] = None
    Platelets: Optional[float] = None
    # ... add more fields as your machine supports them

@app.post("/api/ingest")
async def ingest_machine_data(reading: MachineReading, api_key: str = Depends(verify_api_key)):
    """
    Accept a single machine reading and return an immediate prediction.
    Secured via X-API-Key and logged to the HIPAA audit trail.
    """
    pid = reading.patient_id
    
    # Audit log entry for data access/ingestion
    write_audit_log(pid, "ingest_data", {
        "source": "machine",
        "parameters": reading.model_dump(exclude_none=True, exclude={"patient_id"})
    })
    
    patient = ward.get_patient(pid)

    if patient:
        # Update simulated patient with real values where provided
        updates = reading.model_dump(exclude_none=True, exclude={"patient_id"})
        patient.current_values.update({k: float(v) for k, v in updates.items()})
        patient.history.append({**patient.current_values,
                                "timestamp": datetime.now().isoformat(),
                                "source": "machine"})

        summary = patient.to_summary()
        window = patient.get_feature_window()
    else:
        # Unknown patient — create a temporary summary
        summary = {"patient_id": pid, "name": pid, "age": 60, "gender": "U",
                   "bed": "External", "vitals": {}, "labs": {}}
        vitals_keys = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp"]
        lab_keys    = ["Lactate","Creatinine","WBC","pH","Platelets"]
        d = reading.model_dump(exclude_none=True, exclude={"patient_id"})
        summary["vitals"] = {k: d[k] for k in vitals_keys if k in d}
        summary["labs"]   = {k: d[k] for k in lab_keys if k in d}
        window = None

    prediction = engine.predict(summary, feature_window=window)
    _latest_predictions[pid] = {**summary, **prediction}
    
    # Audit log entry for the prediction run
    write_audit_log(pid, "prediction", {
        "score": prediction.get("score"),
        "level": prediction.get("level"),
        "alert": prediction.get("alert")
    })
    
    # EHR Alert Writeback for high risk thresholds
    if prediction.get("alert") or prediction.get("level") in ["HIGH", "CRITICAL"]:
        await dispatch_ehr_alert(pid, prediction["score"], prediction["level"], prediction.get("evidence", {}))
        
    return prediction



# ── FHIR Observation Integration ─────────────────────────────────────────────
LOINC_MAP = {
    "8867-4": "HR",       # Heart Rate
    "2708-6": "O2Sat",    # SpO2 / Oxygen Saturation
    "2927-4": "Temp",     # Temperature
    "8480-6": "SBP",      # Systolic BP
    "8462-4": "DBP",      # Diastolic BP
    "8478-0": "MAP",      # Mean Arterial Pressure
    "9279-1": "Resp",     # Respiratory Rate
    "2276-4": "Lactate",   # Lactate
    "2160-0": "Creatinine", # Creatinine
    "6690-2": "WBC",      # WBC
    "2744-1": "pH",       # pH
    "26515-7": "Platelets", # Platelets
}

def parse_fhir_observation(resource: dict) -> tuple:
    """Parse a FHIR Observation resource. Returns (patient_id, {feature_name: value})."""
    if not isinstance(resource, dict) or resource.get("resourceType") != "Observation":
        return None, {}
    
    subject = resource.get("subject", {})
    ref = subject.get("reference", "")
    if not ref or "/" not in ref:
        return None, {}
    patient_id = ref.split("/")[-1]
    
    value_qty = resource.get("valueQuantity", {})
    if "value" not in value_qty:
        return patient_id, {}
    val = float(value_qty["value"])
    
    feature = None
    code_data = resource.get("code", {})
    for coding in code_data.get("coding", []):
        code = coding.get("code")
        if code in LOINC_MAP:
            feature = LOINC_MAP[code]
            break
            
    if not feature:
        text = code_data.get("text", "").lower()
        if not text:
            for coding in code_data.get("coding", []):
                if coding.get("display"):
                    text = coding["display"].lower()
                    break
        if text:
            text = text.lower()
            if "heart" in text or "pulse" in text: feature = "HR"
            elif "oxygen" in text or "spo2" in text or "sat" in text: feature = "O2Sat"
            elif "temp" in text: feature = "Temp"
            elif "systolic" in text: feature = "SBP"
            elif "diastolic" in text: feature = "DBP"
            elif "mean arterial" in text or "map" in text: feature = "MAP"
            elif "resp" in text or "breath" in text: feature = "Resp"
            elif "lactate" in text: feature = "Lactate"
            elif "creatinine" in text: feature = "Creatinine"
            elif "wbc" in text or "leukocyte" in text: feature = "WBC"
            elif "ph" in text: feature = "pH"
            elif "platelet" in text: feature = "Platelets"
            
    if not feature:
        return patient_id, {}
        
    return patient_id, {feature: val}

@app.post("/api/fhir/Observation")
async def ingest_fhir_observation(payload: dict, api_key: str = Depends(verify_api_key)):
    """
    Accept FHIR Observation (single resource or Bundle) and trigger prediction updates.
    Secured via X-API-Key and logged to the HIPAA audit trail.
    """
    resources = []
    if payload.get("resourceType") == "Bundle":
        for entry in payload.get("entry", []):
            res = entry.get("resource")
            if res:
                resources.append(res)
    else:
        resources.append(payload)
        
    updates_by_patient = {}
    for res in resources:
        pid, update = parse_fhir_observation(res)
        if pid and update:
            if pid not in updates_by_patient:
                updates_by_patient[pid] = {}
            updates_by_patient[pid].update(update)
            
    if not updates_by_patient:
        raise HTTPException(status_code=400, detail="No parseable FHIR Observations found")
        
    results = {}
    for pid, updates in updates_by_patient.items():
        # Audit log entry for data access/ingestion
        write_audit_log(pid, "ingest_data", {
            "source": "fhir",
            "parameters": updates
        })
        
        patient = ward.get_patient(pid)
        if patient:
            patient.current_values.update({k: float(v) for k, v in updates.items()})
            patient.history.append({
                **patient.current_values,
                "timestamp": datetime.now().isoformat(),
                "source": "fhir"
            })
            summary = patient.to_summary()
            window = patient.get_feature_window()
        else:
            summary = {"patient_id": pid, "name": pid, "age": 60, "gender": "U",
                       "bed": "External", "vitals": {}, "labs": {}}
            vitals_keys = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp"]
            lab_keys    = ["Lactate","Creatinine","WBC","pH","Platelets"]
            summary["vitals"] = {k: updates[k] for k in vitals_keys if k in updates}
            summary["labs"]   = {k: updates[k] for k in lab_keys if k in updates}
            window = None
            
        prediction = engine.predict(summary, feature_window=window)
        _latest_predictions[pid] = {**summary, **prediction}
        results[pid] = prediction
        
        # Audit log entry for the prediction run
        write_audit_log(pid, "prediction", {
            "score": prediction.get("score"),
            "level": prediction.get("level"),
            "alert": prediction.get("alert")
        })
        
        # EHR Alert Writeback for high risk thresholds
        if prediction.get("alert") or prediction.get("level") in ["HIGH", "CRITICAL"]:
            await dispatch_ehr_alert(pid, prediction["score"], prediction["level"], prediction.get("evidence", {}))
            
    return {"status": "success", "results": results}



class TrajectoryUpdate(BaseModel):
    patient_id: str
    trajectory: str

@app.post("/api/simulator/trajectory")
async def update_patient_trajectory(data: TrajectoryUpdate):
    """
    Update a simulated patient's clinical trajectory.
    This lets clinicians test how the system reacts as patient physiology changes.
    """
    pid = data.patient_id
    traj = data.trajectory.lower()
    if traj not in ["stable", "deteriorating", "septic", "recovering"]:
        raise HTTPException(status_code=400, detail="Invalid trajectory type. Choose from: stable, deteriorating, septic, recovering")
        
    patient = ward.get_patient(pid)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {pid} not found")
        
    old_traj = patient.trajectory
    patient.trajectory = traj
    
    # If starting deterioration, shift values towards sepsis thresholds immediately to simulate onset
    if traj in ["septic", "deteriorating"] and old_traj not in ["septic", "deteriorating"]:
        patient.current_values["Lactate"] = max(patient.current_values.get("Lactate", 1.2), 3.2)
        patient.current_values["HR"] = max(patient.current_values.get("HR", 75), 112.0)
        patient.current_values["MAP"] = min(patient.current_values.get("MAP", 82), 61.0)
        patient.current_values["O2Sat"] = min(patient.current_values.get("O2Sat", 97), 90.5)
        patient.current_values["Resp"] = max(patient.current_values.get("Resp", 16), 26.0)
    elif traj == "stable":
        # Reset to normal ranges
        patient.current_values["Lactate"] = 1.2
        patient.current_values["HR"] = 75.0
        patient.current_values["MAP"] = 82.0
        patient.current_values["O2Sat"] = 98.0
        patient.current_values["Resp"] = 16.0
        patient.sepsis_onset_hour = None
        patient.current_values["SepsisLabel"] = 0.0
        
    return {
        "status": "success",
        "patient_id": pid,
        "old_trajectory": old_traj,
        "new_trajectory": traj,
        "message": f"Updated {patient.name} trajectory to {traj.upper()}"
    }


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/monitor")
async def ws_ward_monitor(websocket: WebSocket):
    """Full ward live stream — broadcasts every patient every tick."""
    await websocket.accept()
    _ward_connections.add(websocket)

    # Send immediate snapshot on connect
    summaries = ward.all_summaries()
    init_payload = []
    for s in summaries:
        pred = _latest_predictions.get(s["patient_id"], {})
        init_payload.append({**s, **pred})
    await websocket.send_text(json.dumps({
        "type": "init",
        "tick": _tick_count,
        "timestamp": datetime.now().isoformat(),
        "patients": init_payload,
        "model": engine._get_model_display_name() if engine.model_loaded else "Heuristic",
        "model_type": getattr(engine, "model_type", "heuristic"),
        "tick_interval_s": TICK_INTERVAL_SECONDS,
    }))

    try:
        while True:
            await asyncio.sleep(60)  # keep-alive (data pushed by tick loop)
    except WebSocketDisconnect:
        _ward_connections.discard(websocket)


@app.websocket("/ws/patient/{patient_id}")
async def ws_patient_stream(websocket: WebSocket, patient_id: str):
    """Per-patient live stream."""
    await websocket.accept()
    if patient_id not in _patient_connections:
        _patient_connections[patient_id] = set()
    _patient_connections[patient_id].add(websocket)

    # Send immediate snapshot
    patient = ward.get_patient(patient_id)
    if patient:
        summary = patient.to_summary()
        pred = _latest_predictions.get(patient_id, {})
        await websocket.send_text(json.dumps({
            "type": "patient_init",
            **summary, **pred,
            "history": patient.history[-24:],
        }))

    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        if patient_id in _patient_connections:
            _patient_connections[patient_id].discard(websocket)


# ──────────────────────────────────────────────────────────────────────────────
# Extreme CDSS Endpoints: Dynamic Model Switcher, Counterfactuals, SSC, FHIR
# ──────────────────────────────────────────────────────────────────────────────

class ModelSwitchRequest(BaseModel):
    model_type: str  # "ensemble", "tcn", "transformer", "bilstm_attention", "heuristic"


class InterventionRequest(BaseModel):
    adjustments: Dict[str, float]  # e.g. {"MAP": 75.0, "Lactate": 1.8}


@app.post("/api/model/switch")
async def switch_model_endpoint(req: ModelSwitchRequest):
    """
    Dynamically switch active deep learning architecture without restarting server.
    """
    res = engine.switch_model(req.model_type)
    if res["status"] == "error":
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@app.get("/api/model/status")
async def model_status():
    """Returns active model details and supported architectures."""
    return {
        "active_model_name": engine._get_model_display_name(),
        "active_model_type": getattr(engine, "model_type", "heuristic"),
        "model_loaded": engine.model_loaded,
        "supported_models": [
            {"id": "ensemble", "name": "Deep Sepsis Ensemble (BiLSTM+Transformer+TCN)", "type": "Multi-Model Soft Voting"},
            {"id": "tcn", "name": "Temporal Convolutional Network (TCN)", "type": "Dilated Causal CNN"},
            {"id": "transformer", "name": "Transformer Encoder", "type": "Multi-Head Self-Attention"},
            {"id": "bilstm_attention", "name": "BiLSTM + Multi-Head Attention", "type": "Bidirectional Recurrent"},
            {"id": "heuristic", "name": "Clinical Heuristic Baseline", "type": "Deterministic Rule Engine"},
        ]
    }


@app.post("/api/patient/{patient_id}/simulate-intervention")
async def simulate_patient_intervention(patient_id: str, req: InterventionRequest):
    """
    Simulates clinical 'What-If' interventions (fluids, vasopressors, O2, antibiotics)
    and returns instantaneous baseline vs counterfactual risk comparison.
    """
    patient = ward.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    summary = patient.to_summary()
    window = patient.get_feature_window()
    res = engine.simulate_what_if(summary, req.adjustments, feature_window=window)
    return {"patient_id": patient_id, **res}


@app.get("/api/patient/{patient_id}/protocol")
async def get_patient_protocol(patient_id: str):
    """
    Returns Surviving Sepsis Campaign (SSC) Hour-1 action bundle tailored for this patient.
    """
    patient = ward.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    summary = patient.to_summary()
    pred = _latest_predictions.get(patient_id, {})
    risk_score = pred.get("risk_score", 0.0)
    bundle = engine.get_protocol_bundle(summary, risk_score)
    return {"patient_id": patient_id, **bundle}


@app.get("/api/patient/{patient_id}/fhir")
async def export_patient_fhir(patient_id: str):
    """
    Exports a standard HL7 FHIR R4 Bundle for the patient containing RiskAssessment,
    Observations, and Patient metadata.
    """
    patient = ward.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    summary = patient.to_summary()
    pred = _latest_predictions.get(patient_id, {})
    fhir_bundle = engine.get_fhir_export(summary, pred)
    return JSONResponse(content=fhir_bundle, media_type="application/fhir+json")


# ──────────────────────────────────────────────────────────────────────────────
# Mock EHR Receiver (for production testing & validation)
# ──────────────────────────────────────────────────────────────────────────────
class EHRAlert(BaseModel):
    timestamp: str
    patient_id: str
    sepsis_probability: float
    risk_level: str
    alert_triggered: bool
    clinical_evidence: dict

@app.post("/api/mock-ehr/writeback")
async def mock_ehr_writeback(alert: EHRAlert, api_key: str = Depends(verify_api_key)):
    """
    Mock endpoint that acts as the hospital EHR system.
    Receives alerts from the Sepsis Early Prediction System and logs them.
    """
    print(f"\n[MOCK EHR RECEIVED ALERT] Time: {alert.timestamp}")
    print(f"   Patient ID: {alert.patient_id}")
    print(f"   Risk Level: {alert.risk_level} (Prob: {alert.sepsis_probability:.2f})")
    print(f"   Triggered:  {alert.alert_triggered}")
    print(f"   Evidence:   {list(alert.clinical_evidence.keys())}\n")
    return {"status": "success", "received": alert.patient_id}


# ──────────────────────────────────────────────────────────────────────────────
# Static File Fallback for Dashboard Assets
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/{file_path:path}", include_in_schema=False)
async def serve_static_file(file_path: str):
    target = DASHBOARD_DIR / file_path
    if target.is_file():
        media_type = None
        if file_path.endswith(".js"):
            media_type = "application/javascript"
        elif file_path.endswith(".css"):
            media_type = "text/css"
        elif file_path.endswith(".html"):
            media_type = "text/html"
        elif file_path.endswith(".json"):
            media_type = "application/json"
        return FileResponse(str(target), media_type=media_type)
    raise HTTPException(status_code=404, detail="File not found")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
