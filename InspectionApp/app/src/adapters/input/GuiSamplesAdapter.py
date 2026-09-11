import threading
import traceback
from typing import Optional

from app.src.core.services.SampleCaptureService import SampleCaptureService
from app.src.interfaces.ICamera import ICamera


class GuiSamplesAdapter:
    """
    Input adapter for training sample capture mode.

    Drives ``SampleCaptureService`` in a background loop: waits for a GPIO
    trigger signal (from the PLC/robot arm), then runs a full labeled capture
    cycle, and repeats. This adapter is intentionally thin — all hardware
    coordination is delegated to ``SampleCaptureService``.

    The structure mirrors ``GuiInferenceAdapter`` so both adapters share an
    identical lifecycle API from the web frontend (Abigail).

    Lifecycle (via AppFactory — preferred)::

        samples = factory.create_samples_controller()
        factory.initialize_hardware()       # starts camera — call AFTER create_*
        samples.set_label("ok")             # label written to disk (default: "ok")
        samples.start_loop()                # GPIO-triggered loop in background thread
        frame_bytes = samples.get_preview_frame()
        counts = samples.count_all_labels()
        samples.stop_loop()
        factory.shutdown()

    Schedule Timed Captures (see ``SampleCaptureService``): this adapter only
    decides, per cycle, whether to persist and with which label — all
    window/target/timer logic lives in the service. ``enable_schedule()`` /
    ``disable_schedule()`` / ``get_schedule_status()`` are thin delegates.

    Note:
        ``initialize()`` is a convenience method that calls
        ``camera.initialize_camera()`` directly. When using ``AppFactory``,
        call ``factory.initialize_hardware()`` instead — they are equivalent
        but calling both will reinitialize the camera twice (harmless but wasteful).

    Attributes:
        _service (SampleCaptureService): Service that owns all hardware coordination.
        _camera (ICamera): Camera adapter used only for MJPEG preview streaming.
        _label (str): Current label written to disk on each capture cycle.
        _thread (threading.Thread | None): Background capture loop thread.
        _running (bool): True while the loop is active.
        _last_result (dict[str, str] | None): Paths saved in the last completed cycle.
        _result_lock (threading.Lock): Protects reads/writes of ``_last_result``.
        _cycle_count (int): Total completed capture cycles since loop start.
    """

    _MAX_CONSECUTIVE_ERRORS = 5

    def __init__(self, service: SampleCaptureService, camera: ICamera):
        """
        Args:
            service (SampleCaptureService): Application service handling all
                hardware coordination for sample capture.
            camera (ICamera): Camera adapter used for MJPEG preview streaming.
                This is the same physical camera used inside ``service``; the
                adapter holds a reference only for calling
                ``get_preview_frame_to_HTML()``.
        """
        self._service = service
        self._camera  = camera
        self._label: str = "ok"

        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

        self._last_result: Optional[dict[str, str]] = None
        self._result_lock: threading.Lock = threading.Lock()
        self._cycle_count: int = 0
        self._consecutive_errors: int = 0

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def initialize(self) -> None:
        """
        Initialize the camera hardware and select the first channel for preview.

        Must be called once before ``start_loop()`` or ``get_preview_frame()``.
        When using ``AppFactory``, call ``factory.initialize_hardware()`` instead.

        Returns:
            None
        """
        self._camera.initialize_camera()
        self._service.restore_preview()
        print("[OK] GuiSamplesAdapter: camera initialized.")

    def shutdown(self) -> None:
        """
        Stop the capture loop and release all hardware resources.

        Safe to call even if the loop has not been started.

        Returns:
            None
        """
        self.stop_loop()
        self._camera.close_camera()
        print("[OK] GuiSamplesAdapter: shutdown complete.")

    # =========================================================================
    # Loop control
    # =========================================================================

    def start_loop(self) -> None:
        """
        Start the capture loop in a background daemon thread.

        Each iteration calls ``SampleCaptureService.wait_for_trigger()`` and,
        on a positive signal, ``SampleCaptureService.run_capture_cycle(label)``.

        Returns:
            None
        """
        if self._running:
            print("[WARN] GuiSamplesAdapter: loop is already running.")
            return

        self._running = True
        self._service.reset_interrupt()  # Clear any pending interrupt from the previous stop_loop().
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="SampleCaptureLoop",
        )
        self._thread.start()
        print("[OK] GuiSamplesAdapter: capture loop started.")

    def stop_loop(self) -> None:
        """
        Signal the capture loop to stop and wait for the current cycle to finish.

        Returns:
            None
        """
        if not self._running:
            return

        self._running = False
        self._service.interrupt()  # Unblock any wait_for_input call within ~10 ms.
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

        print("[OK] GuiSamplesAdapter: capture loop stopped.")

    @property
    def is_running(self) -> bool:
        """True if the capture loop thread is active."""
        return self._running

    # =========================================================================
    # Label control
    # =========================================================================

    def set_label(self, label: str) -> None:
        """
        Set the label written to disk for subsequent capture cycles.

        Takes effect on the next cycle; does not interrupt the current one.

        Args:
            label (str): Destination subfolder name, e.g. ``'ok'``, ``'nok'``.

        Returns:
            None
        """
        self._label = label
        print(f"[INFO] GuiSamplesAdapter: label set to '{label}'.")

    # =========================================================================
    # Schedule Timed Captures (thin delegates — all logic lives in the service)
    # =========================================================================

    def enable_schedule(self, images_per_window: int, interval_s: float, target_images: int) -> None:
        """
        Enable Schedule Timed Captures, targeting the currently selected label.

        Args:
            images_per_window (int): Production cycles persisted per window.
            interval_s (float): Seconds between window boundaries.
            target_images (int): Total cycles to persist before auto-pausing.
        """
        self._service.enable_schedule(images_per_window, interval_s, target_images, self._label)

    def disable_schedule(self) -> None:
        """Disable Schedule Timed Captures. Manual capture behavior resumes immediately."""
        self._service.disable_schedule()

    def get_schedule_status(self) -> dict:
        """Return the current Schedule Timed Captures state. See ``SampleCaptureService.get_schedule_status()``."""
        return self._service.get_schedule_status()

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

    # =========================================================================
    # Results and status
    # =========================================================================

    def get_last_result(self) -> dict | None:
        """
        Return the paths saved in the most recently completed capture cycle.

        Thread-safe — safe to call from an HTTP request handler.

        Returns:
            dict | None: ``{channel: absolute_path}`` for the last cycle,
                or ``None`` if no cycle has completed yet.
        """
        with self._result_lock:
            return dict(self._last_result) if self._last_result else None

    def get_status(self) -> dict:
        """
        Return the current operational status.

        Returns:
            dict: Keys: ``running``, ``label``, ``cycle_count``.
        """
        return {
            "running":     self._running,
            "label":       self._label,
            "cycle_count": self._cycle_count,
        }

    def count_samples(self, label: str) -> dict[str, int]:
        """
        Count saved ``.jpg`` images per channel for a given label.

        Delegates to ``SampleCaptureService``.

        Args:
            label (str): Label folder name (e.g. ``'ok'``).

        Returns:
            dict[str, int]: Channel name → number of saved images.
        """
        return self._service.count_samples(label)

    def count_all_labels(self) -> dict[str, dict[str, int]]:
        """
        Count samples for every label that exists under ``train_images_path``.

        Delegates to ``SampleCaptureService``.

        Returns:
            dict[str, dict[str, int]]: label → channel → count.
        """
        return self._service.count_all_labels()

    # =========================================================================
    # Private — loop worker
    # =========================================================================

    def _run_loop(self) -> None:
        """
        Background thread: wait for GPIO trigger → run capture cycle → repeat.

        Error handling mirrors ``GuiInferenceAdapter``: full traceback is printed,
        camera recovery is attempted, and the loop stops after
        ``_MAX_CONSECUTIVE_ERRORS`` consecutive failures. The process stays
        alive for status queries after a critical stop.
        """
        while self._running:
            triggered = self._service.wait_for_trigger()

            if not self._running:
                break

            if not triggered:
                continue  # Timeout is normal — loop and wait again.

            try:
                persist, label = self._service.next_cycle_plan(self._label)
                saved = self._service.run_capture_cycle(label, persist=persist)
                with self._result_lock:
                    self._last_result = saved
                self._cycle_count += 1
                self._consecutive_errors = 0
                if persist:
                    self._service.schedule_record_persisted()

            except Exception as e:
                self._consecutive_errors += 1
                print(
                    f"[ERROR] Capture cycle failed "
                    f"(consecutive failures: {self._consecutive_errors}/"
                    f"{self._MAX_CONSECUTIVE_ERRORS}): {e}"
                )
                print(traceback.format_exc())

                if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    print(
                        f"[CRITICAL] {self._MAX_CONSECUTIVE_ERRORS} consecutive capture "
                        f"failures. Stopping sample capture loop. "
                        f"Restart the process to resume."
                    )
                    self._running = False
                    break

                print("[INFO] Attempting camera recovery...")
                try:
                    self._camera.close_camera()
                    self._camera.initialize_camera()
                    self._service.restore_preview()
                    print("[OK] Camera recovery successful.")
                except Exception as recovery_err:
                    print(f"[ERROR] Camera recovery failed: {recovery_err}")
