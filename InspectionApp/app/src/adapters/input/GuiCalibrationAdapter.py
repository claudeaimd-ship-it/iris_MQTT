import os
import threading
import traceback
from typing import Callable, Optional

from app.src.core.services.CalibrationService import CalibrationService
from app.src.interfaces.ICamera import ICamera


class GuiCalibrationAdapter:
    """
    Input adapter for on-device PaDiM calibration.

    Drives ``CalibrationService`` (sweep and calibration runs) in background
    threads so the Flask web server thread is never blocked. Progress is
    reported via a callback injected by ``IrisServer``.

    This adapter is intentionally thin — it only manages thread lifecycle and
    error recovery. All calibration logic lives in ``CalibrationService``.

    Lifecycle::

        cal = factory.create_calibration_controller()
        factory.initialize_hardware()
        cal.start_sweep(view_configs, image_dirs, backbones_dir, progress_cb=...)
        # poll cal.is_running / cal.get_progress() from /api/calibration/status
        cal.start_calibration(view_configs, image_dirs, backbones_dir, block=9, ...)
        factory.shutdown()

    Attributes:
        _service (CalibrationService): Service that owns all calibration logic.
        _camera (ICamera): Camera adapter for MJPEG preview streaming only.
        _thread (threading.Thread | None): Active background task thread.
        _running (bool): True while a task is executing.
        _progress (dict): Most recent progress snapshot.
        _progress_lock (threading.Lock): Protects reads/writes of ``_progress``.
    """

    _MAX_CONSECUTIVE_ERRORS = 3

    def __init__(self, service: CalibrationService, camera: ICamera) -> None:
        """
        Args:
            service (CalibrationService): Calibration service.
            camera (ICamera): Camera used for MJPEG preview only.
        """
        self._service  = service
        self._camera   = camera

        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._cancel_event: threading.Event = threading.Event()

        self._progress: dict = {"step": 0, "total": 0, "message": "", "done": False, "error": None}
        self._progress_lock: threading.Lock = threading.Lock()

    # =========================================================================
    # Public — status
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """True while a sweep or calibration task is running in the background."""
        return self._running

    def get_progress(self) -> dict:
        """
        Return a snapshot of the latest progress state.

        Returns:
            dict: Keys: ``step`` (int), ``total`` (int), ``message`` (str),
                ``done`` (bool), ``error`` (str | None).
        """
        with self._progress_lock:
            return dict(self._progress)

    def get_preview_frame(self) -> bytes:
        """
        Capture a preview frame for the MJPEG stream.

        Returns:
            bytes: JPEG-encoded frame bytes.
        """
        return self._camera.get_preview_frame_to_HTML()

    # =========================================================================
    # Public — sweep
    # =========================================================================

    def start_sweep(
        self,
        view_configs: list[dict],
        image_dirs: dict[str, dict[str, str]],
        backbones_dir: str,
        blocks: tuple[int, int] | None = None,
        params: dict | None = None,
        done_cb: Callable[[list], None] | None = None,
    ) -> None:
        """
        Start a full block sweep in a background thread.

        Does nothing if a task is already running (caller should check
        ``is_running`` first and return HTTP 409).

        Args:
            view_configs (list[dict]): View configs forwarded to
                ``CalibrationService.run_sweep()``.
            image_dirs (dict[str, dict[str, str]]): Image directories.
            backbones_dir (str): Backbone ONNX directory.
            blocks (tuple[int, int] | None): Block range; defaults to (3, 17).
            params (dict | None): Optional scoring parameters.
            done_cb (Callable[[list], None] | None): Called with the list of
                ``SweepResult`` objects when the sweep completes successfully.
        """
        if self._running:
            return

        self._cancel_event.clear()
        self._set_progress(0, 1, "Sweep starting…", done=False, error=None)
        self._thread = threading.Thread(
            target=self._run_sweep,
            args=(view_configs, image_dirs, backbones_dir, blocks, params, done_cb),
            daemon=True,
        )
        self._running = True
        self._thread.start()

    def cancel(self) -> None:
        """Request cancellation of the active sweep. No-op if nothing is running."""
        self._cancel_event.set()

    # =========================================================================
    # Public — calibration
    # =========================================================================

    def start_calibration(
        self,
        view_configs: list[dict],
        image_dirs: dict[str, dict[str, str]],
        backbones_dir: str,
        blocks: dict[str, int],
        model_path: str,
        params: dict | None = None,
        done_cb: Callable[[list], None] | None = None,
    ) -> None:
        """
        Start a final calibration run in a background thread.

        Args:
            view_configs (list[dict]): View configs.
            image_dirs (dict[str, dict[str, str]]): Image directories.
            backbones_dir (str): Backbone ONNX directory.
            blocks (dict[str, int]): Per-view block mapping produced by the
                sweep (e.g. ``{"front_view_A": 9, "side_view_B": 7}``).
            model_path (str): Output directory for calibration files.
            params (dict | None): Optional scoring parameters.
            done_cb (Callable[[list], None] | None): Called with the list of
                ``CalibrationResult`` when done.
        """
        if self._running:
            return

        self._set_progress(0, 1, "Calibration starting…", done=False, error=None)
        self._thread = threading.Thread(
            target=self._run_calibration,
            args=(view_configs, image_dirs, backbones_dir, blocks, model_path, params, done_cb),
            daemon=True,
        )
        self._running = True
        self._thread.start()

    # =========================================================================
    # Private — background tasks
    # =========================================================================

    def _run_sweep(
        self,
        view_configs: list[dict],
        image_dirs: dict[str, dict[str, str]],
        backbones_dir: str,
        blocks: tuple[int, int] | None,
        params: dict | None,
        done_cb: Callable[[list], None] | None,
    ) -> None:
        """Background thread body for sweep."""
        try:
            results = self._service.run_sweep(
                view_configs=view_configs,
                image_dirs=image_dirs,
                backbones_dir=backbones_dir,
                blocks=blocks,
                params=params,
                progress_cb=self._progress_callback,
                cancel_check=lambda: self._cancel_event.is_set(),
            )
            if results is None:
                # Cancelled by user — not an error.
                self._set_progress(
                    step=self._progress["step"],
                    total=self._progress["total"],
                    message="Sweep cancelled.",
                    done=True,
                    error=None,
                )
                return
            self._set_progress(
                step=self._progress["step"],
                total=self._progress["total"],
                message="Sweep complete.",
                done=True,
                error=None,
            )
            if done_cb:
                done_cb(results)
        except Exception:
            error_msg = traceback.format_exc()
            print(f"[ERROR] GuiCalibrationAdapter sweep failed:\n{error_msg}")
            self._set_progress(
                step=self._progress["step"],
                total=self._progress["total"],
                message="Sweep failed — see logs.",
                done=True,
                error=error_msg,
            )
        finally:
            self._running = False

    def _run_calibration(
        self,
        view_configs: list[dict],
        image_dirs: dict[str, dict[str, str]],
        backbones_dir: str,
        blocks: dict[str, int],
        model_path: str,
        params: dict | None,
        done_cb: Callable[[list], None] | None,
    ) -> None:
        """Background thread body for final calibration."""
        try:
            results = self._service.run_calibration(
                view_configs=view_configs,
                image_dirs=image_dirs,
                backbones_dir=backbones_dir,
                blocks=blocks,
                model_path=model_path,
                params=params,
                progress_cb=self._progress_callback,
            )
            self._set_progress(
                step=self._progress["step"],
                total=self._progress["total"],
                message="Calibration complete.",
                done=True,
                error=None,
            )
            if done_cb:
                done_cb(results)
        except Exception:
            error_msg = traceback.format_exc()
            print(f"[ERROR] GuiCalibrationAdapter calibration failed:\n{error_msg}")
            self._set_progress(
                step=self._progress["step"],
                total=self._progress["total"],
                message="Calibration failed — see logs.",
                done=True,
                error=error_msg,
            )
        finally:
            self._running = False

    # =========================================================================
    # Private — helpers
    # =========================================================================

    def _progress_callback(self, step: int, total: int, message: str) -> None:
        """Forwarded from CalibrationService — stores latest progress state."""
        self._set_progress(step=step, total=total, message=message, done=False, error=None)
        print(f"[CAL] ({step}/{total}) {message}")

    def _set_progress(
        self,
        step: int,
        total: int,
        message: str,
        done: bool,
        error: str | None,
    ) -> None:
        with self._progress_lock:
            self._progress = {
                "step":    step,
                "total":   total,
                "message": message,
                "done":    done,
                "error":   error,
            }
