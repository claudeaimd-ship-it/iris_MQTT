import json
import os
import threading
import traceback
from datetime import datetime
from typing import Optional

import cv2

from app.src.core.models.Part import Part
from app.src.core.services.SequenceExecutor import SequenceExecutor
from app.src.interfaces.ICamera import ICamera

# Rollback switch (2026-07-31): the per-cycle crash-recovery handler now
# reinitializes the camera AND the CSI MUX via SequenceExecutor.recover_hardware()
# (see that method's docstring). Set this to False to revert to the previous,
# camera-only recovery (close_camera()+initialize_camera(), no MUX touch) if the
# new behavior causes issues on real multi-camera hardware. Remove this flag once
# the new recovery has been validated on the Pi and is no longer needed.
_USE_FULL_HARDWARE_RECOVERY = True


class GuiInferenceAdapter:
    """
    Input adapter for production inference mode.

    Drives the SequenceExecutor in a background thread, generates sequential
    part IDs, exposes a live MJPEG preview stream, and makes the last
    inspection result available for the web frontend (Abigail).

    This is the primary entry point for the inspection application in production.
    It replaces the simple CLI loop in ``main.py`` and is designed to be driven
    by an HTTP server rather than terminal input.

    Lifecycle (via AppFactory — preferred)::

        controller = factory.create_inference_controller()
        factory.initialize_hardware()   # starts camera — call AFTER create_*
        controller.start_loop()         # begins inspection loop in background thread
        ...
        controller.pause()              # pauses after current cycle
        controller.resume()             # resumes
        ...
        factory.shutdown()              # stops loop and releases hardware

    Note:
        ``initialize()`` is a convenience method that calls
        ``camera.initialize_camera()`` directly. When using ``AppFactory``,
        call ``factory.initialize_hardware()`` instead — they are equivalent
        but calling both will reinitialize the camera twice (harmless but wasteful).

    Attributes:
        _executor (SequenceExecutor): Fully configured inspection executor.
        _camera (ICamera): Camera used for preview streaming.
        _counter_path (str): Path to the JSON counter file for part ID generation.
        _thread (threading.Thread | None): Background inspection loop thread.
        _running (bool): True while the loop is active.
        _pause_event (threading.Event): Clear to pause, set to resume.
        _last_result (Part | None): Most recently completed Part.
        _result_lock (threading.Lock): Protects reads/writes of _last_result.
        _cycle_count (int): Total completed inspection cycles since start.
        _forced_scrap_counter (int): Counts how many upcoming cycles should be forced into NOK steps.
    """

    def __init__(
        self,
        executor: SequenceExecutor,
        camera: ICamera,
        counter_path: str,
        images_path: str | None = None,
    ):
        """
        Args:
            executor (SequenceExecutor): Fully configured inspection executor.
            camera (ICamera): Camera adapter used for preview streaming.
            counter_path (str): Path to the JSON file that persists the daily
                part ID counter. Created automatically if it does not exist.
            images_path (str | None): Root image directory for this product
                (e.g. ``./data/images/my_part/``). When provided, the adapter
                writes the last captured frame per view to
                ``images_path/latest/{view_name}.jpg`` after every cycle so
                that the web frontend can display them even after a server
                restart. Writes are atomic (temp file + ``os.replace``).
        """
        self._executor     = executor
        self._camera       = camera
        self._counter_path = counter_path
        self._images_path: str | None = images_path

        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._dry_run: bool = False
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()  # Start in the resumed (unblocked) state.

        self._last_result: Optional[Part] = None
        self._result_lock: threading.Lock = threading.Lock()
        self._cycle_count: int = 0
        self._consecutive_errors: int = 0
        self._stopped_reason: Optional[str] = None  # Set when the loop stops itself due to a critical error.

        self._forced_scrap_counter: int = 0  # Counts how many upcoming cycles should be forced into NOK steps.

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def initialize(self) -> None:
        """
        Initialize the camera hardware.

        Must be called once before ``start_loop()`` or ``get_preview_frame()``.

        Returns:
            None
        """
        self._camera.initialize_camera()
        print("[OK] GuiInferenceAdapter: camera initialized.")

    def shutdown(self) -> None:
        """
        Stop the inspection loop and release all hardware resources.

        Safe to call even if the loop has not been started.

        Returns:
            None
        """
        self.stop_loop()
        self._camera.close_camera()
        print("[OK] GuiInferenceAdapter: shutdown complete.")

    # =========================================================================
    # Loop control
    # =========================================================================

    def start_loop(self) -> None:
        """
        Start the inspection loop in a background thread.

        Each iteration generates a unique part ID, calls
        ``SequenceExecutor.run(part_id)``, and stores the result. The loop
        blocks while paused and exits cleanly when ``stop_loop()`` is called.

        Returns:
            None
        """
        if self._running:
            print("[WARN] GuiInferenceAdapter: loop is already running.")
            return

        self._running = True
        self._pause_event.set()
        self._consecutive_errors = 0
        self._stopped_reason = None  # Clear any error from a previous run.
        self._executor.reset_interrupt()  # Clear any pending interrupt from the previous stop_loop().
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="InspectionLoop",
        )
        self._thread.start()
        print("[OK] GuiInferenceAdapter: inspection loop started.")

    def stop_loop(self) -> None:
        """
        Signal the inspection loop to stop and wait for the thread to finish.

        Returns:
            None
        """
        if not self._running:
            return

        self._running = False
        self._pause_event.set()  # Unblock a paused loop so it can exit.
        self._executor.interrupt()  # Unblock any wait_for_input call within ~10 ms.

        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

        print("[OK] GuiInferenceAdapter: inspection loop stopped.")

    def pause(self) -> None:
        """
        Pause the inspection loop after the current cycle completes.

        Returns:
            None
        """
        self._pause_event.clear()
        print("[INFO] GuiInferenceAdapter: loop paused.")

    def resume(self) -> None:
        """
        Resume a paused inspection loop.

        Returns:
            None
        """
        self._pause_event.set()
        print("[INFO] GuiInferenceAdapter: loop resumed.")

    def set_dry_run(self, value: bool) -> None:
        """
        Enable or disable dry-run mode for subsequent inspection cycles.

        When dry-run is active, ``SequenceExecutor.run()`` is called with
        ``dry_run=True`` so NOK GPIO steps (step_number < 0) are skipped.
        Safe to call while the loop is stopped; rejected at the API level
        while running.

        Args:
            value (bool): True to enable dry-run, False to disable.

        Returns:
            None
        """
        self._dry_run = value
        print(f"[INFO] GuiInferenceAdapter: dry-run {'enabled' if value else 'disabled'}.")

    def set_forced_scrap(self, cycles: int = 3) -> None:
        """
        Force the next N inspection cycles to execute NOK steps regardless of ML score.

        When active, the loop will route to NOK steps (step_number < 0) for the next
        `cycles` iterations, and log the overall_status as "{Actual_Result}_SCRAP"
        in the traceability logs. Safe to call while the loop is stopped; rejected
        at the API level while running.

        Args:
            cycles (int): Number of upcoming cycles to force into scrap mode (default: 3).

        Returns:
            None
        """
        self._forced_scrap_counter = cycles
        print(f"[INFO] GuiInferenceAdapter: forced scrap mode set for next {cycles} cycles.")

    @property
    def is_running(self) -> bool:
        """True if the inspection loop thread is active."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """True if the loop is running but currently waiting at a pause point."""
        return self._running and not self._pause_event.is_set()

    # =========================================================================
    # Preview
    # =========================================================================

    def get_preview_frame(self) -> bytes:
        """
        Return a JPEG-encoded preview frame for MJPEG streaming.

        Returns:
            bytes: JPEG bytes ready for multipart HTTP streaming.
        """
        return self._camera.get_preview_frame_to_HTML()

    def get_last_inference_frames(self) -> dict:
        """
        Return the captured frames from the most recent inspection cycle.

        Thread-safe — delegates to SequenceExecutor.get_last_captured_frames().

        Returns:
            dict[str, np.ndarray]: Map of view_name to RGB888 numpy frame.
                Empty if no cycle has completed yet.
        """
        return self._executor.get_last_captured_frames()

    # =========================================================================
    # Results and status
    # =========================================================================

    def get_last_result(self) -> Optional[dict]:
        """
        Return the most recently completed inspection result as a plain dict.

        Thread-safe — safe to call from an HTTP request handler while the
        inspection loop runs in the background.

        Returns:
            dict | None: Serialized Part result, or None if no inspection has
                run yet.
        """
        with self._result_lock:
            if self._last_result is None:
                return None
            return self._serialize_result(self._last_result)

    def get_status(self) -> dict:
        """
        Return the current operational status of the adapter.

        Returns:
            dict: Keys: ``running``, ``paused``, ``cycle_count``, ``forced_scrap_cycles``,
                ``stopped_reason``.
        """
        return {
            "running":     self._running,
            "paused":      self.is_paused,
            "cycle_count": self._cycle_count,
            "forced_scrap_cycles": self._forced_scrap_counter,
            "stopped_reason": self._stopped_reason,
        }

    def get_live_cycle_age_s(self) -> float | None:
        """
        Seconds since the active cycle's trigger was received, or ``None`` if
        no cycle is currently processing (idle waiting for the next trigger,
        or the loop is stopped/paused). Passthrough to ``SequenceExecutor`` —
        used by the in-process cycle watchdog in ``IrisServer.py``.
        """
        return self._executor.get_live_cycle_age_s()

    # =========================================================================
    # Private — latest-frame persistence
    # =========================================================================

    _LATEST_JPEG_QUALITY = 80
    _LATEST_MAX_WIDTH    = 800

    def _save_latest_frames(self, frames: dict) -> None:
        """Persist the most recent captured frame per view to disk.

        Writes each frame to ``images_path/latest/{view_name}.jpg`` using an
        atomic temp-file rename (``os.replace``) so readers never see a
        partially-written file.  Frames are resized to at most
        ``_LATEST_MAX_WIDTH`` pixels wide to keep file size small.

        Args:
            frames (dict[str, np.ndarray]): Map of view_name to RGB888 array.

        Returns:
            None
        """
        if not self._images_path or not frames:
            return
        latest_dir = os.path.join(self._images_path, "latest")
        os.makedirs(latest_dir, exist_ok=True)
        for view_name, frame in frames.items():
            try:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                h, w = bgr.shape[:2]
                if w > self._LATEST_MAX_WIDTH:
                    bgr = cv2.resize(
                        bgr,
                        (self._LATEST_MAX_WIDTH, int(h * self._LATEST_MAX_WIDTH / w)),
                    )
                _, jpeg = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._LATEST_JPEG_QUALITY]
                )
                dest = os.path.join(latest_dir, f"{view_name}.jpg")
                tmp  = dest + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(jpeg.tobytes())
                os.replace(tmp, dest)
            except Exception as exc:
                print(f"[WARN] GuiInferenceAdapter: could not save latest frame "
                      f"'{view_name}': {exc}")

    # =========================================================================
    # Private — loop worker
    # =========================================================================

    _MAX_CONSECUTIVE_ERRORS = 5  # Stop the loop after this many back-to-back failures.

    def _run_loop(self) -> None:
        """
        Background thread: run inspection cycles until ``stop_loop()`` is called.

        Error handling strategy:
        - Any exception in a single cycle is caught, logged with full traceback,
          and a camera recovery is attempted before the next cycle.
        - If ``_MAX_CONSECUTIVE_ERRORS`` failures occur in a row, the loop stops
          itself.  The process remains alive so the web server can still serve
          status pages; restart the process (or use the systemd service) to
          resume inspection.
        """
        while self._running:
            self._pause_event.wait()  # Block here when paused.

            if not self._running:
                break

            part_id = self._generate_part_id()
            try:
                part = self._executor.run(part_id, dry_run=self._dry_run, forced_scrap=(self._forced_scrap_counter > 0))
                if self._forced_scrap_counter > 0:
                    part.forced_scrap = True
                    self._forced_scrap_counter -= 1
                    print(f"[INFO] Cycle '{part_id}' is forced into scrap mode "
                          f"(remaining forced scrap cycles: {self._forced_scrap_counter}).")
                    
                if getattr(part, "system_error_paused", False):
                    print(f"[CRITICAL] Cycle '{part_id}' was aborted due to a system error during execution.")
                    self.pause()  # Pause the loop to prevent further cycles until manual intervention.

                with self._result_lock:
                    self._last_result = part
                self._cycle_count += 1
                self._consecutive_errors = 0  # Reset on success.
                self._save_latest_frames(self._executor.get_last_captured_frames())
            except Exception as e:
                self._consecutive_errors += 1
                print(
                    f"[ERROR] Cycle failed for '{part_id}' "
                    f"(consecutive failures: {self._consecutive_errors}/"
                    f"{self._MAX_CONSECUTIVE_ERRORS}): {e}"
                )
                print(traceback.format_exc())
                print(f"[CRITICAL] Unhandled cycle exception! Pausing station to prevent PLC/Machine desync.")
                self.pause()

                if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    print(
                        f"[CRITICAL] {self._MAX_CONSECUTIVE_ERRORS} consecutive cycle "
                        f"failures. Stopping inspection loop. "
                        f"Restart the process (or the systemd service) to resume."
                    )
                    self._stopped_reason = (
                        f"Camera/hardware initialization error: {self._MAX_CONSECUTIVE_ERRORS} "
                        "consecutive cycle failures. Use the 'Restart Iris' button (top-right) "
                        "to restart the service — Stop/Start the loop will not fix this."
                    )
                    self._running = False
                    break

                # Attempt camera + MUX recovery before the next cycle. Delegated to
                # the executor (not done directly on self._camera here) because the
                # MUX also needs to be reinitialized — a camera-only reset is not
                # always enough on multi-camera CSI setups (see
                # SequenceExecutor.recover_hardware() docstring).
                print("[INFO] Attempting camera/MUX recovery...")
                try:
                    if _USE_FULL_HARDWARE_RECOVERY:
                        self._executor.recover_hardware()
                    else:
                        # Legacy camera-only recovery (pre-2026-07-31), kept as a
                        # rollback path — see _USE_FULL_HARDWARE_RECOVERY above.
                        # Atomic under hw_lock — see recover_hardware() docstring.
                        with self._camera.hw_lock:
                            self._camera.close_camera()
                            self._camera.initialize_camera()
                    print("[OK] Camera/MUX recovery successful.")
                except Exception as recovery_err:
                    print(f"[ERROR] Camera/MUX recovery failed: {recovery_err}")
                finally:
                    # Always resume so the next cycle can run (and consecutive
                    # error counter keeps incrementing toward _MAX_CONSECUTIVE_ERRORS
                    # which exits cleanly and lets systemd restart the process).
                    self.resume()

    # =========================================================================
    # Private — part ID counter
    # =========================================================================

    def _generate_part_id(self) -> str:
        """
        Generate a unique, date-scoped part ID and persist the counter to disk.

        The counter resets to 1 each day. Format: ``YYYYMMDD-NNNN``.

        Returns:
            str: Part ID string, e.g. ``'20260518-0042'``.
        """
        today = datetime.now().strftime("%Y%m%d")
        counter_data: dict = {"date": "", "counter": 0}

        if os.path.isfile(self._counter_path):
            try:
                with open(self._counter_path, "r") as f:
                    counter_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass  # Corrupted file — start fresh.

        if counter_data.get("date") != today:
            counter_data = {"date": today, "counter": 0}

        counter_data["counter"] += 1

        os.makedirs(os.path.dirname(os.path.abspath(self._counter_path)), exist_ok=True)
        with open(self._counter_path, "w") as f:
            json.dump(counter_data, f)

        return f"{today}-{counter_data['counter']:06d}"

    # =========================================================================
    # Private — serialization
    # =========================================================================

    @staticmethod
    def _serialize_result(part: Part) -> dict:
        """
        Convert a Part to a plain dict suitable for JSON serialization.

        Args:
            part (Part): Completed Part object.

        Returns:
            dict: Flat representation of the Part and its view results.
        """
        return {
            "part_id":        part.part_id,
            "model_id":       part.model_id,
            "date_inspected": part.date_inspected.strftime("%Y%m%d_%H%M%S"),
            "duration_s":     round(part.time_inspected, 4) if part.time_inspected is not None else None,
            "overall_status": "ERROR_ABORTED" if getattr(part, "system_error_paused", False) else ("OK" if part.overall_status else "NOK") + ("_SCRAP" if part.forced_scrap else "") + ("_DR" if part.dry_run else ""),
            "piece_detected": part.piece_detected,
            "dry_run":        part.dry_run,
            "failed_channel":       getattr(part, "failed_channel", None),
            "failed_channel_error": getattr(part, "failed_channel_error", None),
            "view_results": [
                {
                    "view_name":      r.view,
                    "classification": "OK" if r.is_ok else "NOK",
                    "score":          float(r.score),
                    "threshold_min":  float(r.threshold_used[0]),
                    "threshold_max":  float(r.threshold_used[1]),
                }
                for r in part.inspection_results
            ],
            "triggers": [
                {
                    "step_number": t.step_number,
                    "direction":   t.direction,
                    "pin":         t.pin,
                    "action":      t.action,
                    "result":      t.result,
                }
                for t in part.triggers
            ],
        }
