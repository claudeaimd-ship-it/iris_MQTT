# =============================================================================
# gunicorn configuration for Iris
#
# Used by setup/iris.service. Run from the InspectionApp/ working directory.
#
# gunicorn reads this file automatically when it is named gunicorn.conf.py
# and is located in the working directory, or when passed with --config.
# =============================================================================

import json

# ---------------------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------------------
# Single worker: IrisState is shared in-process (not across workers).
workers     = 1
threads     = 4
worker_class = "gthread"
bind        = "0.0.0.0:5000"

# Worker timeout: how long the worker can be silent (no heartbeat) before
# gunicorn kills it. Default is 30 s, which is far too short when the
# calibration sweep is running heavy ONNX inference in a background thread
# and the Pi is thermally throttled. Set to 0 (infinite) because:
#  - This server is local/industrial (not internet-facing).
#  - The calibration background thread can legitimately consume the CPU for
#    several minutes per block without touching gunicorn's request pipeline.
timeout     = 0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Suppress HTTP access log — silences the browser's 3-second status-poll
# entries that would otherwise flood the application log file.
accesslog   = "/dev/null"

# gunicorn startup/error messages go to stderr (captured by systemd journal).
errorlog    = "-"
loglevel    = "warning"

# ---------------------------------------------------------------------------
# Worker lifecycle hooks
# ---------------------------------------------------------------------------

def post_fork(server, worker):
    """
    Start TimestampedFileLogger inside each worker after the fork so that
    all application-level print() calls are written to the daily log file
    in addition to being captured by the systemd journal.
    """
    try:
        from app.src.core.utils.TimestampedFileLogger import TimestampedFileLogger

        with open("config/default_values.json", "r", encoding="utf-8") as _f:
            _logs_path = json.load(_f).get("logs_path", "./logs")
    except Exception:
        _logs_path = "./logs"

    _logger = TimestampedFileLogger(_logs_path, suffix="iris")
    _logger.start()
    # Keep a reference on the worker object so it is not garbage-collected.
    worker._iris_logger = _logger
