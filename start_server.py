"""
start_server.py — One-click launcher for the Live Sepsis Prediction System
==========================================================================

Usage:
    python start_server.py

Opens the live ward monitor in your browser automatically.
Render.com: reads $PORT env var for dynamic port binding.
"""
import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Render injects $PORT dynamically; fallback to 8000 for local dev
PORT = int(os.environ.get("PORT", 8000))
IS_LOCAL = PORT == 8000


def open_browser():
    time.sleep(2.0)
    webbrowser.open(f"http://127.0.0.1:{PORT}/live.html")
    print(f"\n  Browser opened -> http://127.0.0.1:{PORT}/live.html")
    print(f"  API docs       -> http://localhost:{PORT}/docs")
    print(f"  Ingest data    -> POST http://localhost:{PORT}/api/ingest\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SEPSIS LIVE PREDICTION SYSTEM -- Starting...")
    print(f"  Port: {PORT}")
    print("="*60)

    # Only open browser when running locally (Render has no display)
    if IS_LOCAL:
        t = threading.Thread(target=open_browser, daemon=True)
        t.start()

    import uvicorn
    from backend.server import app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )

