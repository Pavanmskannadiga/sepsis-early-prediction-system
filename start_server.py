"""
start_server.py — One-click launcher for the Live Sepsis Prediction System
==========================================================================

Usage:
    python start_server.py

Opens the live ward monitor in your browser automatically.
"""
import sys
import time
import webbrowser
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

def open_browser():
    time.sleep(2.0)
    webbrowser.open("http://127.0.0.1:8000/live.html")
    print("\n  Browser opened -> http://127.0.0.1:8000/live.html")
    print("  API docs       -> http://localhost:8000/docs")
    print("  Ingest data    -> POST http://localhost:8000/api/ingest\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SEPSIS LIVE PREDICTION SYSTEM -- Starting...")
    print("="*60)

    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    import uvicorn
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
