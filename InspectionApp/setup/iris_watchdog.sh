#!/usr/bin/env bash
# iris_watchdog.sh — External liveness watchdog for Iris: polls /api/info
# from outside the process, catching a HUNG (not crashed) Iris that the
# in-process cycle watchdog (IrisServer.py) can't detect on its own.
#
# Supports both deployment models, picked automatically on every check:
#   Model A — iris.service (headless, no browser). Detected via
#             `systemctl is-active --quiet iris`. Recovery: kill the hung
#             gunicorn process; systemd's Restart=on-failure brings it back.
#   Model B — iris-browser.desktop / start_iris.sh (kiosk, no systemd unit
#             for Iris). Recovery: kill gunicorn + kiosk browser, relaunch
#             start_iris.sh.
#
# Runs as the same user that owns the Iris process — never needs sudo/root.
#
# Installation (as its own systemd service, alongside iris.service):
#   sudo cp setup/iris_watchdog.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now iris_watchdog
#
# Useful commands:
#   sudo systemctl status  iris_watchdog
#   journalctl -u iris_watchdog -f
#
# Usage (manual/foreground, for testing):
#   bash setup/iris_watchdog.sh

set -uo pipefail  # Deliberately NOT -e: a single failed curl/pgrep must never kill the watchdog loop itself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${IRIS_PORT:-5000}"
URL="http://localhost:${PORT}"

# How often to poll /api/info.
POLL_INTERVAL_S="${IRIS_WATCHDOG_POLL_INTERVAL_S:-10}"
# Consecutive misses required before acting (~30s by default with the
# interval above) — avoids restarting Iris over one transient blip (e.g. a
# gunicorn worker briefly busy under load).
FAILURE_THRESHOLD="${IRIS_WATCHDOG_FAILURE_THRESHOLD:-3}"
CURL_TIMEOUT_S=5

consecutive_failures=0

log() {
    echo "[iris_watchdog] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

kill_gunicorn() {
    pkill -f "gunicorn.*IrisServer" 2>/dev/null || true
}

kill_kiosk_browser() {
    # Only ever targets the kiosk profile/window this project itself
    # launches (see start_iris.sh) — never touches an operator's other
    # browser windows if this happens to run on a shared desktop machine.
    pkill -f "firefox.*iris_firefox_kiosk_profile" 2>/dev/null || true
    pkill -f "chromium.*${URL}" 2>/dev/null || true
    pkill -f "chrome.*${URL}" 2>/dev/null || true
}

recover() {
    log "Iris unreachable at $URL for ${FAILURE_THRESHOLD} consecutive checks — recovering..."

    if systemctl is-active --quiet iris 2>/dev/null; then
        log "Model A (iris.service active): killing the hung process; systemd's Restart=on-failure will bring it back."
        kill_gunicorn
    else
        log "Model B (no iris.service unit active): killing gunicorn + kiosk browser, relaunching via start_iris.sh."
        kill_gunicorn
        kill_kiosk_browser
        nohup bash "$SCRIPT_DIR/start_iris.sh" >> "$HOME/.iris_watchdog_restart.log" 2>&1 &
        disown
    fi
}

log "Started — polling $URL/api/info every ${POLL_INTERVAL_S}s (threshold: ${FAILURE_THRESHOLD} consecutive failures)."

while true; do
    sleep "$POLL_INTERVAL_S"

    if curl -sf --max-time "$CURL_TIMEOUT_S" "$URL/api/info" > /dev/null 2>&1; then
        if [[ $consecutive_failures -gt 0 ]]; then
            log "Iris responded again after ${consecutive_failures} failed check(s) — resetting counter."
        fi
        consecutive_failures=0
    else
        consecutive_failures=$((consecutive_failures + 1))
        log "Check failed (${consecutive_failures}/${FAILURE_THRESHOLD})."
        if [[ $consecutive_failures -ge $FAILURE_THRESHOLD ]]; then
            recover
            consecutive_failures=0
        fi
    fi
done
