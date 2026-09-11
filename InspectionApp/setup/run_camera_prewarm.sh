#!/usr/bin/env bash
# run_camera_prewarm.sh — cold-boot CSI camera prewarm: runs
# scripts/cold_boot_camera_prewarm.py once per channel, each in its own
# process (killed via external process-group timeout if it hangs) — clears a
# kernel-level camera wedge that only a full process kill can fix. Full
# rationale/history: section 24 of .github/copilot-instructions.md.
#
# Best-effort by design: never blocks Iris from starting.
#
# Usage:
#   bash setup/run_camera_prewarm.sh
#   PREWARM_PYTHON=/path/to/venv/bin/python bash setup/run_camera_prewarm.sh
#   (PREWARM_PYTHON defaults to "python" — correct when a venv is already
#   active, as in start_iris.sh; iris.service overrides it to the venv path.)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$APP_DIR"

PREWARM_PYTHON="${PREWARM_PYTHON:-python}"

# Logs whether anything still holds /dev/video0 and whether a libcamera IPA
# proxy is still running — the two things a group-kill is supposed to clear.
# Prints automatically instead of requiring a manual pgrep/lsof check.
log_camera_diagnostics() {
    local label="$1"
    echo "[run_camera_prewarm] --- camera diagnostics ($label) ---"
    if command -v lsof >/dev/null 2>&1; then
        local held
        held="$(lsof /dev/video0 2>/dev/null)"
        if [[ -n "$held" ]]; then
            echo "[run_camera_prewarm] /dev/video0 still held by:"
            echo "$held"
        else
            echo "[run_camera_prewarm] /dev/video0 is free."
        fi
    else
        echo "[run_camera_prewarm] lsof not installed — skipping /dev/video0 check (sudo apt-get install lsof)."
    fi
    local procs
    procs="$(pgrep -af 'rpi/vc4' 2>/dev/null)"
    if [[ -n "$procs" ]]; then
        echo "[run_camera_prewarm] Possible orphaned libcamera/IPA processes:"
        echo "$procs"
    else
        echo "[run_camera_prewarm] No orphaned libcamera/IPA processes found."
    fi
    echo "[run_camera_prewarm] --- end diagnostics ---"
}

# Runs `scripts/cold_boot_camera_prewarm.py "$@"` in its own session (so its
# PID doubles as its process group ID), polls once per second, and
# force-kills the whole group if it is still alive after $timeout_s.
run_prewarm_stage() {
    local timeout_s="$1"
    shift
    setsid "$PREWARM_PYTHON" scripts/cold_boot_camera_prewarm.py "$@" &
    local pid=$!
    local elapsed=0
    while (( elapsed < timeout_s )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            return $?
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "[run_camera_prewarm] '$*' did not finish within ${timeout_s}s — force-killing process group $pid"
    kill -9 -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    log_camera_diagnostics "after killing '$*'"
    return 1
}

# Reverted to the field-validated (4/4 success) timeout budget (Sept 9 2026)
# — see section 24 of .github/copilot-instructions.md for the full history.
PROBE_TIMEOUT_S=10
CHANNEL_TIMEOUT_S=8

echo "[run_camera_prewarm] Stage 1/2: probe (sacrificial, expected to fail quickly on a genuine cold boot)..."
run_prewarm_stage "$PROBE_TIMEOUT_S" probe

# Fixed list (keep in sync with config/default_values.json's csi_channels).
# One short-lived process per channel so a wedge on one channel can't eat
# into another's time budget — see section 24 of copilot-instructions.md.
echo "[run_camera_prewarm] Stage 2/2: per-channel warm-up (A, B, C, D)..."
for channel in A B C D; do
    run_prewarm_stage "$CHANNEL_TIMEOUT_S" channel "$channel"
done

log_camera_diagnostics "final state"
echo "[run_camera_prewarm] Done."
