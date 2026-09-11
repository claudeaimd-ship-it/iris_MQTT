"""
cold_boot_camera_prewarm.py — Rustic, no-threads, two-stage CSI camera
prewarm, run BEFORE gunicorn/Iris starts (both cold boot and plain restart).

Root cause this works around: on a genuine cold boot, the very first
`start_stream()` on some CSI channels reproducibly hits a kernel-level
`videobuf2` "driver bug: stop_streaming operation is leaving buffer 0 in
active state" wedge — surfacing as `Dequeue timer ... has expired!` /
`Camera frontend has timed out!` roughly 1s after the first frame is
requested, even though `Picamera2.start()` itself returned without error.

History (full detail in section 24 of .github/copilot-instructions.md):
disposable-subprocess designs (per-channel, then cumulative-prefix) were
abandoned as erratic. A rustic single-process design followed, mirroring
legacy (src/01-MuestrasPi_InferenciaPi) as literally as possible — but a
bare blocking call can hang forever once the ISP is wedged, so a
thread-based call timeout (`_call_with_timeout`, daemon thread + Event) was
added. Field testing then showed erratic/corrupted preview images on
channel B with that thread-based timeout in place — the thread itself, not
just the hang, is suspected to interfere with the Picamera2 instance. This
version removes ALL threading from this script (100% bare/blocking calls,
exactly like legacy) and moves timeout enforcement OUTSIDE the process
entirely, to the wrapping shell orchestrator (`setup/run_camera_prewarm.sh`),
which force-kills the whole process GROUP (not just this PID) if a stage
does not finish in time — the only mechanism confirmed to reliably clear
the kernel-level wedge.

Two CLI modes, each invoked as its OWN fresh process by
`setup/run_camera_prewarm.sh`, each with its own external timeout, so a
hang on one channel never poisons another channel's attempt or eats into
its time budget:

  probe — sacrificial, expected to fail in well under a minute on a genuine
    cold boot: starts the camera and takes exactly one throwaway capture,
    WITHOUT touching GPIO/I2C or switching channel at all (the MUX is left
    wherever it physically is). Purpose is only to absorb the very first
    post-boot stream-start wedge in a disposable process; the wrapper kills
    it (whole process group) after 10s if it hangs.

  channel <NAME> — warms up exactly one physical channel (e.g. 'A'):
    starts the camera on whatever channel the MUX is currently on, then
    stop() → apply <NAME>'s gpio_state/i2c_cmd → start() → one throwaway
    capture. `setup/run_camera_prewarm.sh` invokes this once per catalog
    channel (A, B, C, D), each as its own short-lived process with its own
    timeout — replacing an earlier design (`run_sweep`, one process walking
    all channels internally) that was abandoned after field testing showed
    a single wedged channel would consume the whole stage's time budget and
    leave every later channel completely untried (see section 24 of
    .github/copilot-instructions.md).

Both modes: best-effort, log `[OK]`/`[WARN]`, never raise. Iris is started
regardless of either stage's outcome (see `setup/run_camera_prewarm.sh`) —
this script's exit code is informational only.

Usage:
    cd /path/to/InspectionApp
    python3 scripts/cold_boot_camera_prewarm.py probe
    python3 scripts/cold_boot_camera_prewarm.py channel A
"""
import json
import os
import sys
import time

from gpiozero import OutputDevice
from picamera2 import Picamera2
from libcamera import controls

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "config")
_DEFAULT_VALUES_PATH = os.path.join(_CONFIG_DIR, "default_values.json")


def _load_defaults() -> dict:
    with open(_DEFAULT_VALUES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _setup_gpio_pins() -> dict:
    # Same close+reopen dance as RpiCsiMuxAdapter.__init__ — releases stale
    # gpiozero handles left behind by a previous process.
    pins = {}
    for bcm in (4, 17, 18):
        pin = OutputDevice(bcm, initial_value=False)
        pin.close()
        pins[bcm] = OutputDevice(bcm, initial_value=False)
    return pins


def _configure_camera(picam: Picamera2, capture_resolution: tuple, preview_resolution: tuple) -> None:
    config = picam.create_video_configuration(
        main={"size": capture_resolution, "format": "RGB888"},
        lores={"size": preview_resolution, "format": "YUV420"},
    )
    picam.configure(config)
    picam.set_controls({"AeEnable": True, "AwbMode": controls.AwbModeEnum.Fluorescent})


def run_probe() -> int:
    """Sacrificial stage: one throwaway capture, no GPIO/I2C, no channel switch."""
    defaults = _load_defaults()
    capture_resolution = tuple(defaults["camera_capture_resolution"])
    preview_resolution = tuple(defaults["camera_preview_resolution"])

    picam = Picamera2()
    try:
        _configure_camera(picam, capture_resolution, preview_resolution)
        picam.start()
        time.sleep(0.02)  # Allow the sensor to stabilize after starting, same as legacy.
        picam.capture_array("lores")
        print("[OK] cold_boot_camera_prewarm (probe): initial capture succeeded.")
        return 0
    except Exception as exc:
        print(f"[WARN] cold_boot_camera_prewarm (probe): initial capture failed: {exc}")
        return 1
    finally:
        try:
            picam.stop()
        except Exception:
            pass
        try:
            picam.close()
        except Exception:
            pass


def run_channel(channel_name: str) -> int:
    """Warms up exactly one physical channel — a fresh process per channel, no threads."""
    defaults = _load_defaults()
    channels = defaults["csi_channels"]
    capture_resolution = tuple(defaults["camera_capture_resolution"])
    preview_resolution = tuple(defaults["camera_preview_resolution"])

    channel = next((c for c in channels if c["name"] == channel_name), None)
    if channel is None:
        print(f"[WARN] cold_boot_camera_prewarm (channel {channel_name}): unknown channel name.")
        return 1

    pins = _setup_gpio_pins()
    picam = Picamera2()
    try:
        _configure_camera(picam, capture_resolution, preview_resolution)
        picam.start()
        time.sleep(0.02)  # Allow the sensor to stabilize after starting, same as legacy.
        picam.stop()
        for bcm, value in channel["gpio_state"].items():
            pins[int(bcm)].value = bool(value)
        os.system(channel["i2c_cmd"])
        picam.start()
        picam.capture_array("lores")
        print(f"[OK] cold_boot_camera_prewarm (channel {channel_name}): warmed up.")
        return 0
    except Exception as exc:
        print(f"[WARN] cold_boot_camera_prewarm (channel {channel_name}): failed: {exc}")
        return 1
    finally:
        try:
            picam.stop()
        except Exception:
            pass
        try:
            picam.close()
        except Exception:
            pass
        for pin in pins.values():
            try:
                pin.close()
            except Exception:
                pass


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "probe":
        return run_probe()
    if len(sys.argv) == 3 and sys.argv[1] == "channel":
        return run_channel(sys.argv[2])
    print("Usage: cold_boot_camera_prewarm.py probe | channel <NAME>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

