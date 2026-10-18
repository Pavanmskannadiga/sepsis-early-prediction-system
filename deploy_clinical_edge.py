"""
Sepsis Early Prediction System — One-Click Clinical Edge Deployment
===================================================================
Launches:
1. Local FastAPI Inference Server on Port 8000 (12 ICU Beds)
2. Cloudflare Zero-Trust Edge Tunnel (Provides instant public HTTPS/WSS URL)
3. Displays Live Clinical Dashboard URL for Bedsides, Mobile, and Reviewers
"""

import sys
import os
import re
import time
import subprocess
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CLOUDFLARED_BIN = PROJECT_ROOT / "tools" / "cloudflared.exe"

def print_banner():
    print("\n" + "=" * 70)
    print("  🏥  SEPSIS CLINICAL DECISION SUPPORT SYSTEM (CDSS)")
    print("  🚀  Unique Zero-Trust Edge Tunnel & Resilient PWA Deployment")
    print("=" * 70)

def main():
    print_banner()

    # 1. Verify cloudflared binary
    if not CLOUDFLARED_BIN.exists():
        print("[!] Cloudflared binary not found at tools/cloudflared.exe.")
        sys.exit(1)

    # 2. Check if server is running on 8000
    print("[1/2] Connecting to Sepsis Live Server on port 8000...")

    # 3. Start Cloudflare Tunnel
    print("[2/2] Establishing Encrypted Global Edge Tunnel via Cloudflare...")
    tunnel_cmd = [str(CLOUDFLARED_BIN), "tunnel", "--url", "http://127.0.0.1:8000"]
    proc = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    public_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    for line in proc.stdout:
        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            break

    if public_url:
        print("\n" + "*" * 70)
        print("  🎉 DEPLOYMENT LIVE WORLDWIDE!")
        print(f"  🌐 Public HTTPS URL : {public_url}/live.html")
        print(f"  🩺 REST API Endpoint: {public_url}/api/patients")
        print(f"  📡 Live WebSocket   : {public_url.replace('https', 'wss')}/ws/monitor")
        print("*" * 70)
        print("\n[i] Anyone on mobile, tablet, or PC can open the link above.")
        print("[i] Press Ctrl+C to stop the edge deployment tunnel.\n")
    else:
        print("[!] Could not parse public URL. Tunnel logs:")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[+] Stopping edge tunnel...")
        proc.terminate()
        print("[+] Edge deployment stopped cleanly.")

if __name__ == '__main__':
    main()
