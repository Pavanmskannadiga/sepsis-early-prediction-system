"""
Clinical Data Ingestion Adapter — HL7 V2 MLLP & FHIR Receiver
=============================================================

This script acts as the clinical gateway between bedside hardware monitors and
the sepsis prediction system. It supports:

1. HL7 MLLP TCP Server (default, port 2575):
   Accepts standard HL7 V2 ORU^R01 (Observation Result) messages, parses patient
   identification (PID) and vital/lab measurements (OBX), pushes predictions to
   the FastAPI server, and returns an HL7 ACK response.

2. Simulation Client (--simulate):
   Acts as a bedside physiological monitor, generating realistic patient data
   and streaming it as HL7 MLLP messages to the adapter.

Usage:
    # Run the HL7 MLLP Server:
    python scripts/clinical_adapter.py --host 0.0.0.0 --port 2575

    # Run the Bedside Simulator (in a separate terminal):
    python scripts/clinical_adapter.py --simulate --target-port 2575
"""

import sys
import time
import socket
import argparse
import random
import json
from datetime import datetime
import urllib.request
import urllib.error

# LOINC clinical code mappings used to map incoming machine streams
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

# Inverse map for simulator to construct clinical HL7 codes
INV_LOINC = {v: k for k, v in LOINC_MAP.items()}

# MLLP Protocol constants
VT = b'\x0b'  # Vertical Tab (Start of block)
FS = b'\x1c'  # File Separator (End of block)
CR = b'\x0d'  # Carriage Return

# ──────────────────────────────────────────────────────────────────────────────
# HL7 V2 Parser & Generator
# ──────────────────────────────────────────────────────────────────────────────

def parse_hl7_oru_message(msg_str: str) -> tuple:
    """
    Parse a standard HL7 V2 ORU^R01 message.
    Returns (patient_id, message_id, {feature_name: value})
    """
    patient_id = None
    message_id = "MSG00001"
    features = {}

    # HL7 segments are delimited by carriage return (\r), sometimes normalized to \n
    segments = [s.strip() for s in msg_str.replace('\n', '\r').split('\r') if s.strip()]

    for seg in segments:
        fields = seg.split('|')
        if not fields:
            continue
        
        seg_type = fields[0]

        if seg_type == "MSH":
            # Message header
            if len(fields) > 9:
                message_id = fields[9]
        
        elif seg_type == "PID":
            # Patient identification
            if len(fields) > 3:
                # PID-3 contains Patient ID. Frequently formatted as ID^^^Assigner
                pid_field = fields[3]
                patient_id = pid_field.split('^')[0]
        
        elif seg_type == "OBX":
            # Observation segment
            if len(fields) > 5:
                # OBX-3 is Code, OBX-5 is Value
                code_field = fields[3]
                val_field = fields[5]
                
                # Extract LOINC code or display text
                code_parts = code_field.split('^')
                loinc_code = code_parts[0]
                display_name = code_parts[1].lower() if len(code_parts) > 1 else ""

                try:
                    val = float(val_field)
                except ValueError:
                    continue  # skip non-numeric fields

                # Determine mapped feature
                feature = None
                if loinc_code in LOINC_MAP:
                    feature = LOINC_MAP[loinc_code]
                else:
                    # Fallback to string matching display name
                    if "heart" in display_name or "pulse" in display_name: feature = "HR"
                    elif "oxygen" in display_name or "spo2" in display_name: feature = "O2Sat"
                    elif "temp" in display_name: feature = "Temp"
                    elif "systolic" in display_name: feature = "SBP"
                    elif "diastolic" in display_name: feature = "DBP"
                    elif "map" in display_name or "mean arterial" in display_name: feature = "MAP"
                    elif "resp" in display_name: feature = "Resp"
                    elif "lactate" in display_name: feature = "Lactate"
                    elif "creatinine" in display_name: feature = "Creatinine"
                    elif "wbc" in display_name: feature = "WBC"
                    elif "ph" in display_name: feature = "pH"
                    elif "platelet" in display_name: feature = "Platelets"
                
                if feature:
                    features[feature] = val

    return patient_id, message_id, features


def generate_hl7_ack(message_id: str, success: bool, note: str = "") -> str:
    """Generate a standard HL7 ACK message response."""
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    status = "AA" if success else "AE"  # Application Accept vs Application Error
    ack = (
        f"MSH|^~\\&|SEPSIS_SYSTEM|HOSPITAL|MONITOR_ADAPTER|BEDSIDE_MONITOR|{ts}||ACK^R01^ACK|{message_id}|P|2.3\r"
        f"MSA|{status}|{message_id}|{note}\r"
    )
    return ack


def make_mllp_frame(msg_str: str) -> bytes:
    """Wrap a message string in MLLP protocol bytes."""
    return VT + msg_str.encode("utf-8") + FS + CR


# ──────────────────────────────────────────────────────────────────────────────
# REST Forwarder
# ──────────────────────────────────────────────────────────────────────────────

def forward_to_api(api_url: str, patient_id: str, readings: dict) -> dict:
    """Forward parsed readings to the FastAPI /api/ingest endpoint."""
    import os
    url = f"{api_url}/api/ingest"
    payload = {"patient_id": patient_id, **readings}
    data_bytes = json.dumps(payload).encode("utf-8")
    
    api_key = os.getenv("SEPSIS_API_KEY", "sepsis_dev_secret_key_123")
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=3.0) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data
    except Exception as e:
        print(f"  [ERROR] Failed to post to API: {e}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Server Mode (MLLP Listener)
# ──────────────────────────────────────────────────────────────────────────────

def run_mllp_server(host: str, port: int, api_url: str):
    """Run a TCP server to accept HL7 messages via MLLP protocol."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print("\n" + "="*70)
        print(f"  CLINICAL HL7 ADAPTER SERVER STARTED")
        print(f"  Listening on MLLP TCP : {host}:{port}")
        print(f"  Forwarding API        : {api_url}")
        print("="*70 + "\n")
    except Exception as e:
        print(f"Failed to bind HL7 MLLP server to {host}:{port}: {e}")
        sys.exit(1)

    while True:
        try:
            conn, addr = server_socket.accept()
            print(f"  [CONN] Connected bedside monitor at {addr[0]}:{addr[1]}")
            
            # Read socket data
            data_buffer = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data_buffer += chunk
                
                # Check if we have complete MLLP frames
                if VT in data_buffer and FS + CR in data_buffer:
                    start_idx = data_buffer.find(VT)
                    end_idx = data_buffer.find(FS + CR)
                    
                    if start_idx < end_idx:
                        # Extract the raw HL7 message content
                        raw_msg = data_buffer[start_idx + 1:end_idx].decode("utf-8", errors="ignore")
                        print(f"  [HL7] Received message ({len(raw_msg)} chars)")
                        
                        # Parse HL7
                        pid, msg_id, readings = parse_hl7_oru_message(raw_msg)
                        
                        if pid:
                            print(f"  [PARSE] Patient: {pid} | Vitals: {readings}")
                            # Forward to prediction engine
                            api_res = forward_to_api(api_url, pid, readings)
                            if api_res:
                                score = api_res.get("risk_score", 0.0)
                                level = api_res.get("risk_level", "LOW")
                                print(f"  [MODEL] Prediction: Risk={score:.3f} ({level})")
                            
                            # Respond with HL7 ACK
                            ack_msg = generate_hl7_ack(msg_id, success=True, note="Processed by Antigravity Gateway")
                            conn.sendall(make_mllp_frame(ack_msg))
                            print(f"  [ACK] Sent MLLP ACK for Message ID: {msg_id}")
                        else:
                            print("  [WARN] Failed to extract Patient ID from PID segment")
                            ack_msg = generate_hl7_ack(msg_id, success=False, note="Missing Patient ID")
                            conn.sendall(make_mllp_frame(ack_msg))

                        # Clean buffer
                        data_buffer = data_buffer[end_idx + 2:]
            
            conn.close()
        except KeyboardInterrupt:
            print("\nShutting down HL7 Server...")
            break
        except Exception as e:
            print(f"  [ERROR] MLLP Server error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Bedside Monitor Simulation Mode
# ──────────────────────────────────────────────────────────────────────────────

def run_bedside_simulator(target_host: str, target_port: int, interval: float, target_pid: str):
    """Simulate a continuous HL7 stream from ICU bedside monitor."""
    print("\n" + "="*70)
    print(f"  BEDSIDE MONITOR SIMULATOR STARTED")
    print(f"  Streaming to HL7 MLLP server : {target_host}:{target_port}")
    print(f"  Interval                     : {interval}s")
    print("="*70 + "\n")

    pids = [target_pid] if target_pid else ["P008401", "P008402", "P008403"]
    names = {
        "P008401": "Sarah Jenkins",
        "P008402": "Robert Miller",
        "P008403": "David Vance"
    }
    
    # Track simulated physiological states
    states = {}
    for pid in pids:
        # Some normal, some deteriorating
        states[pid] = {
            "HR": 75.0, "O2Sat": 98.5, "Temp": 37.0,
            "SBP": 120.0, "MAP": 85.0, "DBP": 75.0, "Resp": 15.0,
            "Lactate": 1.1, "Creatinine": 0.9, "WBC": 8.0,
            "septic_drift": (pid == "P008401")  # Jenkins will deteriorate
        }

    msg_seq = 1000

    while True:
        for pid in pids:
            # Shift physiology slightly
            st = states[pid]
            drift = st["septic_drift"]
            
            if drift:
                # Slowly drift to sepsis
                st["HR"] += random.uniform(-0.5, 2.5)
                st["O2Sat"] -= random.uniform(-0.1, 0.4)
                st["Temp"] += random.uniform(-0.02, 0.15)
                st["SBP"] -= random.uniform(-1.0, 3.0)
                st["MAP"] -= random.uniform(-0.5, 1.8)
                st["Resp"] += random.uniform(-0.1, 1.0)
                if random.random() < 0.15:
                    st["Lactate"] += random.uniform(0.1, 0.5)
                    st["Creatinine"] += random.uniform(0.05, 0.2)
            else:
                # Standard physiological noise
                st["HR"] += random.uniform(-1.5, 1.5)
                st["O2Sat"] = min(100.0, max(94.0, st["O2Sat"] + random.uniform(-0.2, 0.2)))
                st["Temp"] += random.uniform(-0.05, 0.05)
                st["SBP"] += random.uniform(-2.0, 2.0)
                st["MAP"] += random.uniform(-1.0, 1.0)
                st["Resp"] += random.uniform(-0.5, 0.5)

            # Clamp physiologically plausible values
            st["HR"] = min(180.0, max(45.0, st["HR"]))
            st["O2Sat"] = min(100.0, max(75.0, st["O2Sat"]))
            st["Temp"] = min(41.5, max(34.0, st["Temp"]))
            st["SBP"] = min(200.0, max(60.0, st["SBP"]))
            st["MAP"] = min(140.0, max(40.0, st["MAP"]))
            st["Resp"] = min(40.0, max(6.0, st["Resp"]))
            
            # Format HL7 ORU message
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            msg_seq += 1
            msg_id = f"MSG{msg_seq:05d}"
            p_name = names.get(pid, "Bedside Patient")

            # MSH segment
            hl7_lines = [
                f"MSH|^~\\&|BEDSIDE_MONITOR|ICU_WARD|SEPSIS_SYSTEM|HOSPITAL|{ts}||ORU^R01|{msg_id}|P|2.3",
                f"PID|1||{pid}^^^MRN||{p_name}|||M",
                f"PV1|1|I|ICU-BED-{pid[-2:]}|||||||||||||||||||||||||||||||||||||||{ts}"
            ]
            
            # OBX segments for vital signs and key labs
            obx_idx = 1
            for k, val in st.items():
                if k == "septic_drift" or k not in INV_LOINC:
                    continue
                code = INV_LOINC[k]
                # Format float values nicely
                fmt_val = f"{val:.1f}" if k not in ["Lactate", "Creatinine"] else f"{val:.2f}"
                hl7_lines.append(f"OBX|{obx_idx}|NM|{code}^{k}^LN||{fmt_val}||||||F")
                obx_idx += 1

            hl7_msg = "\r".join(hl7_lines) + "\r"
            
            # Send to MLLP listener
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((target_host, target_port))
                
                # Send frame
                sock.sendall(make_mllp_frame(hl7_msg))
                print(f"  [SEND] Sent HL7 message for patient {pid} ({p_name})")
                
                # Wait for ACK response
                response_data = b""
                while True:
                    chunk = sock.recv(1024)
                    if not chunk:
                        break
                    response_data += chunk
                    if FS + CR in response_data:
                        break
                
                if response_data:
                    start_idx = response_data.find(VT)
                    end_idx = response_data.find(FS + CR)
                    if start_idx != -1 and end_idx != -1:
                        ack_str = response_data[start_idx+1:end_idx].decode("utf-8", errors="ignore")
                        ack_lines = ack_str.split("\r")
                        msa = [l for l in ack_lines if l.startswith("MSA")]
                        if msa:
                            print(f"  [ACK] Received: {msa[0]}")
                
                sock.close()
            except Exception as e:
                print(f"  [ERROR] Bedside simulator failed to connect or send: {e}")

        # Sleep till next reading interval
        time.sleep(interval)


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Data Ingestion Adapter — HL7 / FHIR")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host for HL7 MLLP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=2575, help="Binding port for HL7 MLLP server (default: 2575)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of FastAPI sepsis server (default: http://localhost:8000)")
    parser.add_argument("--simulate", action="store_true", help="Run in Bedside Monitor Simulation client mode")
    parser.add_argument("--simulate-patient", default="", help="Specific patient ID to simulate (optional)")
    parser.add_argument("--simulate-interval", type=float, default=5.0, help="Simulation update frequency in seconds (default: 5.0)")
    
    args = parser.parse_args()

    try:
        if args.simulate:
            run_bedside_simulator(
                target_host="127.0.0.1" if args.host == "0.0.0.0" else args.host,
                target_port=args.port,
                interval=args.simulate_interval,
                target_pid=args.simulate_patient
            )
        else:
            run_mllp_server(host=args.host, port=args.port, api_url=args.api_url)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
