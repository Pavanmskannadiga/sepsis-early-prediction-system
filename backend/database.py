"""
Database Module for Sepsis Early Prediction System
===================================================
Provides persistent local SQLite storage for:
- Clinical user accounts and authentication credentials
- Patient demographics and bed assignments
- Real-time ICU telemetry streams (vital signs & lab values)
- AI model prediction history and SHAP explanation logs
- HIPAA clinical audit trail and intervention logs

Uses Python's standard library sqlite3 for zero-configuration, cross-platform portability.
"""

import os
import json
import sqlite3
import hashlib
import secrets
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import config

DB_PATH = config.DATA_DIR / "sepsis_clinical.db"


def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hashes a password with a cryptographic salt using SHA-256."""
    if not salt:
        salt = secrets.token_hex(16)
    salted = f"{salt}${password}".encode("utf-8")
    pwd_hash = hashlib.sha256(salted).hexdigest()
    return f"{salt}${pwd_hash}", salt


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verifies a password against the stored salt$hash string."""
    try:
        parts = stored_hash.split("$")
        if len(parts) != 2:
            return False
        salt, expected_hash = parts
        salted = f"{salt}${password}".encode("utf-8")
        calc_hash = hashlib.sha256(salted).hexdigest()
        return secrets.compare_digest(calc_hash, expected_hash)
    except Exception:
        return False


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes database schema and seeds initial administrative/clinical staff."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            )
        """)

        # 2. Patients Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                bed TEXT NOT NULL,
                trajectory TEXT NOT NULL,
                admission_time TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            )
        """)

        # 3. Telemetry Stream Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vitals_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                icu_hours REAL NOT NULL,
                hr REAL,
                o2sat REAL,
                temp REAL,
                sbp REAL,
                map REAL,
                dbp REAL,
                resp REAL,
                lactate REAL,
                creatinine REAL,
                wbc REAL,
                ph REAL,
                platelets REAL,
                raw_json TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)

        # 4. Predictions & Inference Logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                model_used TEXT NOT NULL,
                alert_triggered INTEGER DEFAULT 0,
                clinical_scores TEXT,
                top_features TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)

        # 5. HIPAA Clinical Audit Trail
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT,
                action TEXT NOT NULL,
                user_name TEXT,
                details TEXT
            )
        """)

        # Indexes for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_pid ON vitals_telemetry(patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON vitals_telemetry(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_pid ON predictions(patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_pid ON clinical_audit_log(patient_id)")

        # Seed initial users if table is empty
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            initial_staff = [
                ("admin", "admin123", "Chief Medical Information Officer", "admin"),
                ("doctor", "sepsis2026", "Dr. Pavana M.S. (Attending Intensivist)", "intensivist"),
                ("resident", "resident2026", "Dr. Megharaja S. (ICU Fellow)", "resident"),
                ("nurse", "icu-nurse", "Charge Nurse Srujan", "nurse"),
                ("researcher", "research", "Clinical Researcher Abubakkar", "researcher"),
            ]
            now_iso = datetime.now().isoformat()
            for u, p, fn, r in initial_staff:
                pwd_hash, _ = _hash_password(p)
                cursor.execute(
                    "INSERT INTO users (username, password_hash, full_name, role, created_at, is_active) VALUES (?, ?, ?, ?, ?, 1)",
                    (u, pwd_hash, fn, r, now_iso)
                )
            conn.commit()
            print("[Database] Initialized SQLite clinical database and seeded 5 default staff accounts.")


# ──────────────────────────────────────────────────────────────────────────────
# Authentication & User Management API
# ──────────────────────────────────────────────────────────────────────────────
def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Validates user credentials against the database."""
    username = username.strip().lower()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE LOWER(username) = ? AND is_active = 1", (username,))
        row = cursor.fetchone()
        if not row:
            return None
        
        if _verify_password(password, row["password_hash"]):
            return {
                "id": row["id"],
                "username": row["username"],
                "full_name": row["full_name"],
                "role": row["role"],
                "created_at": row["created_at"],
            }
    return None


def create_user(username: str, password: str, full_name: str, role: str) -> Tuple[bool, str]:
    """Adds a new clinical user to the database."""
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password cannot be empty"
    
    pwd_hash, _ = _hash_password(password)
    now_iso = datetime.now().isoformat()
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, full_name, role, created_at, is_active) VALUES (?, ?, ?, ?, ?, 1)",
                (username, pwd_hash, full_name, role, now_iso)
            )
            conn.commit()
            return True, f"User {username} created successfully"
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists"
    except Exception as e:
        return False, str(e)


def list_users() -> List[Dict[str, Any]]:
    """Returns all registered users (excluding password hashes)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, full_name, role, created_at, is_active FROM users ORDER BY id ASC")
        return [dict(row) for row in cursor.fetchall()]


# ──────────────────────────────────────────────────────────────────────────────
# Clinical Telemetry & Prediction Logging
# ──────────────────────────────────────────────────────────────────────────────
def upsert_patient(summary: Dict[str, Any]):
    """Records or updates patient demographics in the database."""
    now_iso = datetime.now().isoformat()
    pid = summary.get("patient_id")
    if not pid:
        return
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patients (patient_id, name, age, gender, bed, trajectory, admission_time, updated_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(patient_id) DO UPDATE SET
                bed = excluded.bed,
                trajectory = excluded.trajectory,
                updated_at = excluded.updated_at
        """, (
            pid,
            summary.get("name", "Unknown"),
            summary.get("age", 50),
            summary.get("gender", "M"),
            summary.get("bed", "ICU-00"),
            summary.get("trajectory", "stable"),
            summary.get("admission_time", now_iso),
            now_iso
        ))
        conn.commit()


def log_telemetry_and_prediction(summary: Dict[str, Any], prediction: Dict[str, Any]):
    """Stores one hourly telemetry reading and model prediction into SQLite."""
    pid = summary.get("patient_id")
    if not pid:
        return
    v = summary.get("vitals", {})
    l = summary.get("labs", {})
    now_iso = datetime.now().isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # 1. Log telemetry
        cursor.execute("""
            INSERT INTO vitals_telemetry (
                patient_id, timestamp, icu_hours, hr, o2sat, temp, sbp, map, dbp, resp,
                lactate, creatinine, wbc, ph, platelets, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pid, now_iso, summary.get("icu_hours", 0.0),
            v.get("HR"), v.get("O2Sat"), v.get("Temp"), v.get("SBP"), v.get("MAP"), v.get("DBP"), v.get("Resp"),
            l.get("Lactate"), l.get("Creatinine"), l.get("WBC"), l.get("pH"), l.get("Platelets"),
            json.dumps({"vitals": v, "labs": l})
        ))

        # 2. Log prediction
        cursor.execute("""
            INSERT INTO predictions (
                patient_id, timestamp, risk_score, risk_level, model_used, alert_triggered,
                clinical_scores, top_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pid, now_iso,
            prediction.get("risk_score", 0.0),
            prediction.get("risk_level", "LOW"),
            prediction.get("model_used", "BiLSTM+TCN Hybrid"),
            1 if (prediction.get("alert") or prediction.get("risk_level") in ["HIGH", "CRITICAL"]) else 0,
            json.dumps(prediction.get("clinical_scores", {})),
            json.dumps(prediction.get("top_features", []))
        ))
        conn.commit()


def log_audit_event(action: str, patient_id: Optional[str] = None, user_name: Optional[str] = "system", details: Optional[Dict] = None):
    """Logs a HIPAA audit trail event into the SQLite database."""
    now_iso = datetime.now().isoformat()
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO clinical_audit_log (timestamp, patient_id, action, user_name, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                now_iso, patient_id, action, user_name, json.dumps(details or {})
            ))
            conn.commit()
    except Exception as e:
        print(f"[Database Audit Error] {e}")


def get_database_statistics() -> Dict[str, Any]:
    """Retrieves high-level metadata and row counts from SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM patients")
        total_patients = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM vitals_telemetry")
        total_telemetry = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM clinical_audit_log")
        total_audits = cursor.fetchone()[0]

    db_size_bytes = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    return {
        "status": "connected",
        "database_file": str(DB_PATH.name),
        "database_size_kb": round(db_size_bytes / 1024, 2),
        "total_users": total_users,
        "total_patients": total_patients,
        "total_telemetry_readings": total_telemetry,
        "total_predictions": total_predictions,
        "total_audit_events": total_audits,
        "last_checked": datetime.now().isoformat()
    }


def query_patient_telemetry(patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves historical telemetry rows for a patient."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, patient_id, timestamp, icu_hours, hr, o2sat, temp, sbp, map, dbp, resp, lactate, creatinine, wbc, ph, platelets
            FROM vitals_telemetry
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (patient_id, limit))
        return [dict(row) for row in cursor.fetchall()]


def query_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent HIPAA audit trail events."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, patient_id, action, user_name, details
            FROM clinical_audit_log
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = []
        for r in cursor.fetchall():
            d = dict(r)
            try:
                d["details"] = json.loads(d["details"])
            except Exception:
                pass
            rows.append(d)
        return rows
