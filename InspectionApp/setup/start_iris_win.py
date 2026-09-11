#!/usr/bin/env python3
"""
start_iris_win.py — Windows launcher for the Iris server.

Usage:
    python setup\\start_iris_win.py

Run from the InspectionApp root directory (or via Iris_win.bat, which does
this automatically). Windows equivalent of setup/start_iris.sh.

gunicorn cannot run on Windows — it relies on os.fork()/fcntl, which are
POSIX-only. This launcher uses waitress instead (pure Python, no native
dependencies). Host, port, and logging behave the same as the Linux path:
  - host 0.0.0.0, port 5000 (override with the IRIS_PORT environment variable)
  - logs_path read from config/default_values.json, suffix "iris"
    (see app/src/core/utils/TimestampedFileLogger.py)
"""

import json
import os
import sys
import threading
import time
import urllib.request
import webbrowser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_VALUES_PATH = os.path.join(APP_DIR, "config", "default_values.json")

PORT = int(os.environ.get("IRIS_PORT", "5000"))
URL = f"http://localhost:{PORT}"
MAX_WAIT_S = 30


def _read_logs_path() -> str:
    try:
        with open(DEFAULT_VALUES_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("logs_path", "./logs")
    except Exception:
        return "./logs"


def _wait_and_open_browser() -> None:
    """Poll the server until it responds, then open it in the default browser."""
    print(f"[start_iris_win] Waiting for server at {URL} ...")
    deadline = time.monotonic() + MAX_WAIT_S
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/api/info", timeout=1):
                print("[start_iris_win] Server ready.")
                webbrowser.open(URL)
                return
        except Exception:
            time.sleep(1)
    print(f"[start_iris_win] WARNING: server did not respond within {MAX_WAIT_S}s — open {URL} manually.")


def main() -> int:
    os.chdir(APP_DIR)
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)

    import waitress

    from app.src.core.utils.TimestampedFileLogger import TimestampedFileLogger
    from iris.IrisServer import create_iris_app

    logs_path = _read_logs_path()
    logger = TimestampedFileLogger(logs_path, suffix="iris")
    logger.start()

    threading.Thread(target=_wait_and_open_browser, daemon=True).start()

    try:
        print(f"[start_iris_win] Starting Iris server on {URL} (waitress, 4 threads) ...")
        app = create_iris_app()
        waitress.serve(app, host="0.0.0.0", port=PORT, threads=4)
    finally:
        logger.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
