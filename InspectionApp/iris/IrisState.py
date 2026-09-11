import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.src.AppFactory import AppFactory
    from app.src.adapters.input.GuiInferenceAdapter import GuiInferenceAdapter
    from app.src.adapters.input.GuiSamplesAdapter import GuiSamplesAdapter
    from app.src.adapters.input.GuiCalibrationAdapter import GuiCalibrationAdapter


class IrisState:
    """
    Thread-safe application state for the Iris web server.

    Holds the single AppFactory instance, the active controller, and any
    transient state needed by the builder and inspection pages. All access
    to mutable fields must go through the ``lock`` context manager.

    Attributes:
        factory (AppFactory | None): Active AppFactory. None until a sequence
            is loaded.
        controller (GuiInferenceAdapter | GuiSamplesAdapter | None): Active
            input adapter. None until a factory method has been called.
        current_mode (str): ``"inference"`` or ``"samples"``.
        dry_run (bool): When True, NOK steps are skipped in the next inspection
            cycle. Persists across Start/Stop cycles until explicitly toggled.
            Rejected (409) while loop is running.
        current_sequence_path (str | None): Absolute path to the active
            sequence JSON, or None.
        captured_frames (dict[str, bytes]): Most recent JPEG bytes per camera
            channel (e.g. ``{"A": <bytes>, "B": <bytes>}``). Updated by
            on-demand capture and by the inspection loop between cycles.
        draft (dict | None): Sequence draft currently being edited in the
            builder. None when no builder session is active.
        draft_path (str): Path where the draft is persisted between page
            reloads.
        calibration_controller (GuiCalibrationAdapter | None): Active
            calibration adapter for the sweep / fit pipeline. None when not
            running a sweep or calibration task.
        cal_capture_controller (GuiSamplesAdapter | None): Active samples
            adapter used exclusively by the calibration capture loop. Kept
            separate from ``controller`` so the inspection loop guard
            (``is_running``) is not triggered by calibration captures.
            None when the calibration capture loop is not running.
        calibration_target (str): Current capture target for calibration image
            collection: ``"train_ok"``, ``"test_ok"``, or ``"test_nok"``.
        calibration_progress (dict | None): Latest progress snapshot from the
            background calibration task. Keys: ``step``, ``total``,
            ``message``, ``done``, ``error``. None when no task has run yet.
        sweep_results (list): Latest sweep results from the last completed
            sweep. Each entry is a serializable dict derived from
            ``SweepResult``.
        best_blocks (dict[str, int]): Best MobileNetV2 block per view, as
            determined by the last completed sweep (highest sep_ratio per
            view). Populated by ``_on_sweep_done`` in IrisServer. Empty until
            a sweep has been run. Used by ``run_calibration`` so the operator
            never has to select blocks manually.
        controller_stopping (bool): True from the moment ``api_stop`` begins
            calling ``ctrl.stop_loop()`` until the background thread has fully
            joined. During this window ``controller._running`` is already
            ``False`` but the thread is still alive and owns the camera.
            Routes that reinitialise hardware must check ``is_busy`` instead
            of ``is_running`` to avoid touching the camera while the thread
            is still tearing down.
        lock (threading.Lock): Coarse-grained lock. Acquire before reading or
            writing any field. Do NOT hold across blocking I/O.
    """

    def __init__(self, draft_path: str = "config/sequence_draft.json") -> None:
        """
        Args:
            draft_path (str): File path where the builder draft is auto-saved.
        """
        self.factory: "AppFactory | None" = None
        self.controller: "GuiInferenceAdapter | GuiSamplesAdapter | None" = None
        self.current_mode: str = "inference"
        self.dry_run: bool = False
        self.current_sequence_path: str | None = None
        self.captured_frames: dict[str, bytes] = {}
        self.draft: dict | None = None
        self.draft_path: str = draft_path
        self.calibration_controller: "GuiCalibrationAdapter | None" = None
        self.cal_capture_controller: "GuiSamplesAdapter | None" = None
        self.calibration_target: str = "train_ok"
        self.calibration_progress: dict | None = None
        self.sweep_results: list = []
        self.best_blocks: dict[str, int] = {}
        self.controller_stopping: bool = False
        self.lock: threading.Lock = threading.Lock()

    # ── Convenience helpers ───────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True if a controller loop is currently active."""
        if self.controller is None:
            return False
        return getattr(self.controller, "_running", False)

    @property
    def is_busy(self) -> bool:
        """True if the controller is running OR is in the process of stopping.

        Use this instead of ``is_running`` for any guard that precedes a
        hardware reinitialisation (``_load_sequence``, ``initialize_hardware``,
        ``factory.shutdown``). Between ``api_stop`` setting ``_running=False``
        and the background thread actually joining, ``is_running`` returns
        ``False`` while the camera is still owned by the dying thread.
        ``is_busy`` covers that window via ``controller_stopping``.
        """
        return self.is_running or self.controller_stopping

    @property
    def has_sequence(self) -> bool:
        """True if a sequence is loaded in the factory."""
        return self.factory is not None and self.current_sequence_path is not None

    def store_frame(self, channel: str, jpeg_bytes: bytes) -> None:
        """Store the latest JPEG capture for a camera channel (no lock needed for writes from a single thread)."""
        self.captured_frames[channel] = jpeg_bytes

    def get_frame(self, channel: str) -> bytes | None:
        """Return the last captured JPEG for a channel, or None."""
        return self.captured_frames.get(channel)
