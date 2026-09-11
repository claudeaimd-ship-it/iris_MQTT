#!/usr/bin/env bash
# start_iris.sh — Start the Iris server and open the browser.
#
# Usage:
#   bash setup/start_iris.sh
#
# Run from the InspectionApp root directory. Registers the gunicorn process,
# waits until the server responds, and then opens Firefox in kiosk-style mode
# (falls back to Chromium/Chrome if Firefox is not installed).
#
# To run automatically on boot, add an XDG autostart entry:
#   cp setup/iris-browser.desktop ~/.config/autostart/iris-browser.desktop
#
# Or register as a systemd user service:
#   cp setup/iris.service /etc/systemd/system/
#   systemctl daemon-reload && systemctl enable --now iris

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${IRIS_PORT:-5000}"
URL="http://localhost:${PORT}"
MAX_WAIT=30   # seconds to wait for server to become ready

cd "$APP_DIR"

# Populate the list of virtual environments to check for Python and gunicorn.
VENVS_LIST=("visredPC" "visred" "mezt")
ACTIVATE_VENV=false

# Sweep through the list of virtual environments
echo "[start_iris] Checking for virtual environments: ${VENVS_LIST[*]}"
for VENV in "${VENVS_LIST[@]}"; do
    echo "[start_iris] Checking $VENV... command: source $HOME/venvs/$VENV/bin/activate"
    if [[ -f "$HOME/venvs/$VENV/bin/activate" ]]; then
        echo "[start_iris] Found virtual environment: $VENV"
        source "$HOME/venvs/$VENV/bin/activate"
        ACTIVATE_VENV=true
        break
    else
        echo "[start_iris] No activate script in $VENV, skipping..."
    fi
done

# If no virtual environment was activated, print a warning but continue.
if ! $ACTIVATE_VENV; then
    echo "[start_iris] WARNING: No virtual environment found. Ensure dependencies are installed globally or in one of the expected venvs."
    echo "[start_iris] Exit in 5 seconds..."
    sleep 5
    exit 1
fi  

# If virtual environment was activated.

# Open the browser immediately on a local loading page instead of waiting on
# $URL — camera prewarm + gunicorn boot take ~40s+, and opening straight to
# $URL left the kiosk staring at the desktop the whole time. loading.html
# polls /api/info itself and redirects once the real server is ready.
LOADING_URL="file://$SCRIPT_DIR/loading.html?port=$PORT"
if command -v firefox > /dev/null 2>&1; then
    BROWSER="firefox"
elif command -v firefox-esr > /dev/null 2>&1; then
    BROWSER="firefox-esr"
elif command -v chromium-browser > /dev/null 2>&1; then
    BROWSER="chromium-browser"
elif command -v chromium > /dev/null 2>&1; then
    BROWSER="chromium"
elif command -v google-chrome > /dev/null 2>&1; then
    BROWSER="google-chrome"
else
    BROWSER=""
fi

if [[ ! -s "$SCRIPT_DIR/loading.html" ]]; then
    # Guard against an empty/corrupted loading.html silently showing a blank
    # screen forever — fall back to opening the real URL once ready instead.
    echo "[start_iris] WARNING: loading.html missing or empty — will open $URL directly once ready."
    LOADING_URL="$URL"
    OPEN_BROWSER_LATER=true
else
    OPEN_BROWSER_LATER=false
fi

if [[ -n "$BROWSER" ]]; then
    echo "[start_iris] Opening loading page with $BROWSER ..."
    if [[ "$BROWSER" == "firefox" || "$BROWSER" == "firefox-esr" ]]; then
        FIREFOX_PROFILE_DIR="$HOME/.iris_firefox_kiosk_profile"
        mkdir -p "$FIREFOX_PROFILE_DIR"
        cat > "$FIREFOX_PROFILE_DIR/user.js" <<'EOF'
// Iris kiosk profile — suppress crash/restore/first-run/default-browser
// prompts so nothing but the Iris UI is ever shown on the display.
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.startup.page", 0);
user_pref("browser.tabs.warnOnClose", false);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);
user_pref("browser.aboutwelcome.enabled", false);
user_pref("browser.newtabpage.enabled", false);
EOF
        if ! $OPEN_BROWSER_LATER; then
            "$BROWSER" -kiosk -profile "$FIREFOX_PROFILE_DIR" -new-instance "$LOADING_URL" &
        fi
    else
        if ! $OPEN_BROWSER_LATER; then
            "$BROWSER" \
                --noerrdialogs \
                --disable-infobars \
                --disable-session-crashed-bubble \
                --disable-translate \
                --no-first-run \
                "$LOADING_URL" &
        fi
    fi
else
    echo "[start_iris] No Firefox/Chromium/Chrome found — skipping loading page."
fi

echo "[start_iris] Running cold-boot camera prewarm..."
# See setup/run_camera_prewarm.sh for the two-stage (probe/sweep), no-threads,
# external-process-group-kill design (Sept 9 2026) and why it replaced the
# earlier in-process thread-timeout + bash retry-loop approach.
bash "$SCRIPT_DIR/run_camera_prewarm.sh"

echo "[start_iris] Starting Iris server..."
gunicorn --config gunicorn.conf.py "iris.IrisServer:create_iris_app()" &
GUNICORN_PID=$!
echo "[start_iris] gunicorn PID: $GUNICORN_PID"

# Wait until the server responds — kept for logging/exit-code purposes only;
# the loading page opened above already redirects itself once ready.
echo "[start_iris] Waiting for server at $URL ..."
elapsed=0
until curl -sf "$URL/api/info" > /dev/null 2>&1; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [[ $elapsed -ge $MAX_WAIT ]]; then
        echo "[start_iris] ERROR: server did not start within ${MAX_WAIT}s."
        exit 1
    fi
done
echo "[start_iris] Server ready after ${elapsed}s."

if $OPEN_BROWSER_LATER && [[ -n "$BROWSER" ]]; then
    echo "[start_iris] Opening $URL with $BROWSER (loading.html was unavailable) ..."
    if [[ "$BROWSER" == "firefox" || "$BROWSER" == "firefox-esr" ]]; then
        "$BROWSER" -kiosk -profile "$FIREFOX_PROFILE_DIR" -new-instance "$URL" &
    else
        "$BROWSER" \
            --noerrdialogs \
            --disable-infobars \
            --disable-session-crashed-bubble \
            --disable-translate \
            --no-first-run \
            "$URL" &
    fi
fi

wait "$GUNICORN_PID"
