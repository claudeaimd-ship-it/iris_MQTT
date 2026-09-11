#!/usr/bin/env bash
# install_units.sh — installs iris.service and iris_watchdog.service for the
# CURRENT user, substituting __IRIS_USER__/__IRIS_HOME__ automatically.
#
# Do NOT use systemd's %h specifier instead — it silently broke these units
# in production (see .github/copilot-instructions.md, search "iris.service").
#
# Usage (run as the operator user, NOT root):
#   bash setup/install_units.sh
#
# Then, depending on which deployment option (see
# ../../commands_first_instalation_rpi.txt for the full decision):
#   sudo systemctl enable --now iris             # Option 1 (headless) only
#   sudo systemctl enable --now iris_watchdog    # either option (recommended)
#
# Does NOT touch iris-browser.desktop (Option 2 / kiosk autostart) — that
# file needs no substitution. Just: cp setup/iris-browser.desktop ~/.config/autostart/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_USER="$(whoami)"
REAL_HOME="$HOME"

if [[ "$REAL_USER" == "root" ]]; then
    echo "ERROR: run this as the operator user (e.g. 'pi'), not as root/sudo —" >&2
    echo "       the substituted User=/WorkingDirectory= would otherwise become root." >&2
    exit 1
fi

echo "Installing systemd units for user: $REAL_USER (home: $REAL_HOME)"

for unit in iris.service iris_watchdog.service; do
    sed -e "s#__IRIS_USER__#${REAL_USER}#g" \
        -e "s#__IRIS_HOME__#${REAL_HOME}#g" \
        "$SCRIPT_DIR/$unit" | sudo tee "/etc/systemd/system/$unit" > /dev/null
    echo "  -> /etc/systemd/system/$unit"
done

sudo systemctl daemon-reload
echo "Done. Units installed but NOT enabled/started yet — choose ONE:"
echo "  sudo systemctl enable --now iris             # Option 1 (headless)"
echo "  sudo systemctl enable --now iris_watchdog    # recommended, either option"
