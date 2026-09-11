import json
import os
import subprocess
import threading
import time

from app.src.adapters.output.NoOpMuxAdapter import NoOpMuxAdapter
from app.src.adapters.output.LocalStorageAdapter import LocalStorageAdapter
from app.src.adapters.input.GuiInferenceAdapter import GuiInferenceAdapter
from app.src.adapters.input.GuiSamplesAdapter import GuiSamplesAdapter
from app.src.core.services.InspectionService import InspectionService
from app.src.core.services.SampleCaptureService import SampleCaptureService
from app.src.core.services.SequenceExecutor import SequenceExecutor
from app.src.core.settings.SequenceSettings import SequenceSettings
from app.src.interfaces.IControllableCamera import IControllableCamera
from app.src.interfaces.ICamera import ICamera
from app.src.interfaces.ICsiMux import ICsiMux
from app.src.interfaces.IGpio import IGpio


class AppFactory:
    """
    Composition root for the InspectionApp.

    Reads ``default_values.json`` and the sequence JSON, instantiates all
    adapters and domain services, wires their dependencies, and returns a
    ready-to-use controller.

    Hardware components (camera, MUX, GPIO, repository) are built once and
    shared across all factory methods, so multiple calls to
    ``create_sequence_executor()``, ``create_inference_controller()``, or
    ``create_samples_controller()`` reuse the same hardware instances.

    This is the only place in the application that knows about concrete
    adapter classes. All other layers depend only on interfaces.

    Note:
        Call ``initialize_hardware()`` after any ``create_*`` call to start
        the camera. Call ``shutdown()`` in a finally block to release GPIO pins.
    """

    def __init__(self, default_values_path: str, sequence_path: str):
        """
        Args:
            default_values_path (str): Path to ``config/default_values.json``.
            sequence_path (str): Path to the active sequence JSON file
                (e.g. ``config/sequence_001.json``).
        """
        with open(default_values_path, "r") as f:
            self._defaults: dict = json.load(f)

        with open(sequence_path, "r") as f:
            self._sequence: dict = json.load(f)

        # Load camera capability catalog. Resolved at construction time so
        # _build_hardware() can use it without file I/O on every call.
        catalog_path = os.path.join(
            os.path.dirname(os.path.abspath(default_values_path)), "camera_catalog.json"
        )
        if os.path.isfile(catalog_path):
            with open(catalog_path, "r") as f:
                self._camera_catalog: dict = json.load(f)
        else:
            print(
                f"[WARN] camera_catalog.json not found at '{catalog_path}'. "
                "Using default camera capabilities (supports_af_motor=True)."
            )
            self._camera_catalog = {}

        # Shared hardware — built once by _build_hardware(), reused by all factory methods.
        self._camera:     ICamera | None    = None
        self._mux:        ICsiMux | None    = None
        self._gpio:       IGpio | None      = None
        self._repository: LocalStorageAdapter | None = None

        # Per-channel health status from the last preview attempt (None = healthy).
        # Never sticky: overwritten on every capture_preview_frame() call.
        self._channel_status: dict[str, str | None] = {}

        # Latched alarm: set once automatic recovery gives up entirely (the shared
        # Picamera2 instance is stuck at the OS level). Never auto-cleared — only a
        # real process restart (which recreates this AppFactory) can resolve it.
        self._hardware_wedged_message: str | None = None

    # =========================================================================
    # Private — hardware construction (idempotent)
    # =========================================================================

    def _build_hardware(self) -> None:
        """
        Instantiate the shared camera, MUX, GPIO, and repository adapters.

        Idempotent: does nothing if hardware has already been built.

        Returns:
            None
        """
        if self._camera is not None:
            return

        # Validate number_of_cameras against the declared camera_port list.
        camera_ports: list[str] = self._sequence["hardware"]["camera_port"]
        n_declared: int = self._sequence["hardware"].get("number_of_cameras", len(camera_ports))
        if len(camera_ports) != n_declared:
            raise ValueError(
                f"Sequence declares number_of_cameras={n_declared} but camera_port has "
                f"{len(camera_ports)} entries: {camera_ports}. These must match exactly."
            )

        # Resolve camera-model-specific capabilities from the catalog.
        camera_type: str  = self._sequence["hardware"].get("camera_type", "CSI")
        camera_model: str = self._sequence["hardware"].get("camera_model", "")
        supports_af_motor: bool = True  # Default: assume AF motor present if model not found.
        catalog_models: list[dict] = self._camera_catalog.get(camera_type, [])
        for entry in catalog_models:
            if entry["name"] == camera_model:
                supports_af_motor = entry.get("supports_af_motor", True)
                break
        else:
            if camera_model:
                print(
                    f"[WARN] Camera model '{camera_model}' (type '{camera_type}') not found in "
                    f"camera_catalog.json. Using default capabilities (supports_af_motor=True)."
                )
        
        # Load hardware configuration from the sequence JSON, with fallbacks to defaults.
        capture_resolution = tuple(self._sequence["hardware"].get("camera_capture_resolution"))
        if not capture_resolution or capture_resolution == (0, 0):
            capture_resolution = tuple(self._defaults["camera_capture_resolution"])

        # Per-sequence preview resolution takes priority over the global default.
        preview_res_raw    = (
            self._sequence["hardware"].get("camera_preview_parameters", {}).get("resolution")
            or self._defaults.get("camera_preview_resolution")
        )
        preview_resolution = tuple(preview_res_raw)
        csi_channels       = self._defaults["csi_channels"]
        gpio_configuration = self._sequence["hardware"]["gpio_configuration"]
        preview_exposure   = self._sequence["hardware"]["camera_preview_parameters"]["exposure_time"]
        traceability_path  = self._sequence["paths"]["traceability_inference_path"]
        inference_img_path = self._sequence["paths"].get("inference_images_path")

        # ── Platform selection ──────────────────────────────────────────────
        # device_type defaults to "RaspberryPi" so existing sequences without
        # the field continue to work without modification. Comparison is
        # normalized to handle legacy values like "Raspberry Pi 4B" as well.
        device_type  = self._sequence["hardware"].get("device_type", "RaspberryPi")
        _is_rpi      = "raspberry" in device_type.lower()
        camera_index = self._sequence["hardware"].get("camera_index", 0)

        # ── Camera adapter ─────────────────────────────────────────────────
        if camera_type == "CSI":
            from app.src.adapters.output.CsiCameraAdapter import CsiCameraAdapter
            # Shared with RpiCsiMuxAdapter (via camera.hw_lock) so a channel switch
            # can never interleave with a capture from another thread.
            self._camera = CsiCameraAdapter(
                capture_resolution=capture_resolution,
                preview_resolution=preview_resolution,
                preview_time_exposure=preview_exposure,
                supports_af_motor=supports_af_motor,
                hw_lock=threading.RLock(),
            )
        else:
            # USB cameras and any future types (IP, GStreamer, …) go here.
            from app.src.adapters.output.UsbCameraAdapter import UsbCameraAdapter
            self._camera = UsbCameraAdapter(
                capture_resolution=capture_resolution,
                preview_resolution=preview_resolution,
                camera_index=camera_index,
            )

        # ── MUX adapter ────────────────────────────────────────────────────
        # CSI MUX only makes sense for multi-camera CSI on RPi hardware.
        if camera_type == "CSI" and len(camera_ports) > 1:
            from app.src.adapters.output.RpiCsiMuxAdapter import RpiCsiMuxAdapter
            self._mux = RpiCsiMuxAdapter(
                channels=csi_channels,
                camera=self._camera,
            )
        else:
            self._mux = NoOpMuxAdapter()

        # ── GPIO adapter ───────────────────────────────────────────────────
        # Use real GPIO on RPi; null-object on PC/Jetson until hardware-specific
        # adapters are implemented for those platforms.
        if _is_rpi:
            from app.src.adapters.output.RpiGpioAdapter import RpiGpioAdapter
            self._gpio = RpiGpioAdapter(gpio_configuration=gpio_configuration)
        else:
            from app.src.adapters.output.NullGpioAdapter import NullGpioAdapter
            self._gpio = NullGpioAdapter(gpio_configuration=gpio_configuration)

        self._repository = LocalStorageAdapter(
            traceability_path=traceability_path,
            inference_images_path=inference_img_path,
        )

    # =========================================================================
    # Private — inference engine construction
    # =========================================================================

    def _build_inference_engine(self, model_path: str):
        """
        Instantiate and load the inference engine, selected by model file presence.

        Auto-detection priority (checked by scanning ``model_path``):

        1. ``padim_*_params.npz`` present → ``PaDiMInferenceAdapter``
           (calibration-based, no GPU, runs on Pi or PC)
        2. ``student_*_int8_edgetpu.tflite`` present, or ``inference_device``
           contains ``"Coral"`` → ``CoralUsbInferenceAdapter``
        3. Fallback → ``CpuInferenceAdapter`` (Teacher-Student float32, CPU)

        File-presence checks take priority over the ``inference_device`` sequence
        field so that deploying new model files is sufficient to switch engines
        without editing the sequence JSON.

        Args:
            model_path (str): Directory containing the model files.

        Returns:
            IInferenceEngine: Loaded inference engine.
        """
        inference_device: str = self._sequence["hardware"].get("inference_device", "")

        has_padim = False
        has_coral_student = False
        if os.path.isdir(model_path):
            files = os.listdir(model_path)
            has_padim         = any(
                f.startswith("padim_") and f.endswith("_params.npz") for f in files
            )
            has_coral_student = any(
                f.startswith("student_") and f.endswith("_int8_edgetpu.tflite") for f in files
            )

        if has_padim:
            from app.src.adapters.output.PaDiMInferenceAdapter import PaDiMInferenceAdapter
            engine = PaDiMInferenceAdapter()
        elif has_coral_student or "coral" in inference_device.lower():
            from app.src.adapters.output.CoralUsbInferenceAdapter import CoralUsbInferenceAdapter
            engine = CoralUsbInferenceAdapter()
        else:
            from app.src.adapters.output.CpuInferenceAdapter import CpuInferenceAdapter
            engine = CpuInferenceAdapter()

        engine.load_models_from_directory(model_path)
        return engine

    # =========================================================================
    # Public — lifecycle
    # =========================================================================

    def initialize_hardware(self) -> None:
        """
        Start the camera hardware.

        Must be called once after any ``create_*`` call and before the first
        capture or preview. Hardware must already be built (call a factory
        method first).

        For USB cameras, ``UsbCameraAdapter.initialize_camera()`` reads back the
        resolution that the V4L2 driver actually accepted (which may differ from
        the requested value) and updates ``_capture_resolution`` in place. After
        this call, ``get_capture_resolution()`` returns the real frame dimensions.

        Returns:
            None
        """
        if self._camera is None:
            raise RuntimeError("Hardware not built yet. Call a create_*() method first.")
        self._camera.initialize_camera()
        # self._log_power_status("initialize_hardware")  # diagnostic, disabled (kept for reactivation)

    def get_capture_resolution(self) -> tuple[int, int]:
        """
        Return the actual capture resolution used by the camera adapter.

        For USB cameras the V4L2 driver may silently round the requested
        resolution to the nearest supported sensor mode. Call this **after**
        ``initialize_hardware()`` to obtain the real frame dimensions rather
        than the value configured in the sequence JSON.

        Returns:
            tuple[int, int]: ``(width, height)`` in pixels.
        """
        if self._camera is not None and hasattr(self._camera, "_capture_resolution"):
            return self._camera._capture_resolution  # type: ignore[attr-defined]
        
        capture_resolution = tuple(self._sequence["hardware"].get("camera_capture_resolution"))
        
        if not capture_resolution or capture_resolution == (0, 0):
            capture_resolution_default = tuple(self._defaults["camera_capture_resolution"])
            return capture_resolution_default
        
        return capture_resolution
    
    def shutdown(self) -> None:
        """
        Release all hardware resources gracefully.

        Should be called in a ``finally`` block when the application exits.
        Safe to call even if hardware was never initialized.

        Returns:
            None
        """
        if self._camera is not None:
            self._camera.close_camera()
        if self._mux is not None:
            self._mux.close()
        if self._gpio is not None:
            self._gpio.close()
        self._camera     = None
        self._mux        = None
        self._gpio       = None
        self._repository = None
        self._channel_status = {}
        self._hardware_wedged_message = None

    def reload_sequence(self, sequence_path: str) -> None:
        """
        Swap the active sequence and release all current hardware.

        Call this when the user selects a different product sequence from the
        web interface. The caller is responsible for stopping any active loop
        **before** calling this method. After this call, all hardware
        references are ``None`` — call a ``create_*`` method followed by
        ``initialize_hardware()`` before using any controller.

        Args:
            sequence_path (str): Path to the new sequence JSON file.

        Returns:
            None
        """
        self.shutdown()
        with open(sequence_path, "r") as f:
            self._sequence = json.load(f)
        print(f"[OK] AppFactory: sequence reloaded from '{sequence_path}'.")

    # =========================================================================
    # Public — factory methods
    # =========================================================================

    def create_sequence_executor(self) -> SequenceExecutor:
        """
        Build all inference components and return a ``SequenceExecutor``.

        Loads the inference models and thresholds from the path defined in the
        sequence JSON. Use this for CLI testing; for production use
        ``create_inference_controller()`` instead.

        Returns:
            SequenceExecutor: Fully configured executor ready to call ``run()``.
        """
        self._build_hardware()

        model_path = self._sequence["paths"]["model_path"]
        timeout_ms = self._defaults.get("timeout_waiting_for_signal_ms", 8000)

        # thresholds.json is generated after training and placed in the models directory.
        # Format: {"front_view_section_1_A": {"min": 0.0001, "max": 0.0050}, ...}
        thresholds_path = os.path.join(model_path, "thresholds.json")
        if os.path.isfile(thresholds_path):
            with open(thresholds_path, "r") as f:
                thresholds = json.load(f)
        else:
            print(f"[WARN] thresholds.json not found at '{thresholds_path}'. "
                  f"Inference will raise KeyError when comparing scores.")
            thresholds = {}

        inference_engine = self._build_inference_engine(model_path)

        settings = SequenceSettings(
            sequence=self._sequence,
            thresholds=thresholds,
            eval_config=inference_engine.eval_config,
        )

        inspection_service = InspectionService(
            icamera=self._camera,
            iinference_engine=inference_engine,
            irepository=self._repository,
            settings=settings,
        )

        return SequenceExecutor(
            gpio=self._gpio,
            mux=self._mux,
            camera=self._camera,
            inspection_service=inspection_service,
            sequence=self._sequence,
            default_timeout_ms=timeout_ms,
        )

    def create_inference_controller(self) -> GuiInferenceAdapter:
        """
        Build all components and return a ``GuiInferenceAdapter`` for web-driven
        production inference.

        Returns:
            GuiInferenceAdapter: Ready to call ``initialize()`` then ``start_loop()``.
        """
        executor     = self.create_sequence_executor()
        counter_path = self._defaults.get("part_id_counter_path", "./config/counter.json")
        images_path  = (
            self._sequence["paths"].get("images_path")
            or self._sequence["paths"].get("train_images_path")
        )
        return GuiInferenceAdapter(
            executor=executor,
            camera=self._camera,
            counter_path=counter_path,
            images_path=images_path,
        )

    def create_samples_controller(self) -> GuiSamplesAdapter:
        """
        Build camera / MUX / GPIO components and return a ``GuiSamplesAdapter``
        for training data capture.

        Inference models are intentionally not loaded — this factory method is
        lightweight and only initializes the hardware needed for capturing.

        Returns:
            GuiSamplesAdapter: Ready to call ``initialize()``, ``set_label()``,
                then ``start_loop()``.
        """
        self._build_hardware()

        spotlight_pins    = self._sequence["hardware"].get("spotlight_gpio_pins", [])
        #camera_channels   = self._sequence["hardware"]["camera_port"]
        sequence_steps   = self._sequence.get("steps", [])
        # Support both the new unified key and the legacy key for backward
        # compatibility with sequences created before the path consolidation.
        images_path = (
            self._sequence["paths"].get("images_path")
            or self._sequence["paths"].get("train_images_path", "./data/images/")
        )
        trigger_pin       = self._sequence["hardware"]["trigger_input_pin"]
        timeout_ms        = self._defaults.get("timeout_waiting_for_signal_ms", 8000)

        service = SampleCaptureService(
            camera=self._camera,
            mux=self._mux,
            gpio=self._gpio,
            spotlight_pins=spotlight_pins,
            sequence_steps=sequence_steps,
            trigger_pin=trigger_pin,
            trigger_timeout_ms=timeout_ms,
            images_path=images_path,
            preprocessing_params=self._sequence.get("preprocessing_image_parameters", []),
        )

        return GuiSamplesAdapter(
            service=service,
            camera=self._camera,
        )

    # =========================================================================
    # Public — Iris web interface helpers
    # =========================================================================

    def get_camera_ports(self) -> list[str]:
        """
        Return the list of camera port identifiers declared in the sequence.

        Used by IrisServer to iterate over channels for preview capture
        without accessing private sequence data directly.

        Returns:
            list[str]: Camera port identifiers (e.g. ``["A", "B"]``).
        """
        return list(self._sequence["hardware"]["camera_port"])

    def capture_preview_frame(self, channel: str, exposure_time: int | None = None, lens_position: float | None = None, settle_delay: float = 0.0) -> "numpy.ndarray":
        """
        Select a camera channel via the MUX and capture one full-resolution frame.

        Intended for the builder and inspection on-demand preview. Does NOT
        run the full preprocessing pipeline — returns a raw RGB888 numpy array
        that the caller converts to JPEG.

        Updates ``_channel_status[channel]`` on every call (never sticky): ``None``
        on success, the error message on failure. A failure also triggers a
        best-effort, targeted camera/MUX recovery so a single disconnected channel
        does not leave the shared Picamera2 instance stuck for the other channels.

        Args:
            channel (str): Camera channel identifier (e.g. ``"A"``).
            exposure_time (int | None): Optional exposure time to set for the preview frame.
            lens_position (float | None): Optional lens position to set for the preview frame.
            settle_delay (float): Forwarded to ``ICsiMux.select_channel()``. ``0.0``
                (default) for regular preview calls; only the diagnostic warm-up
                pass (``warmup_all_channels()``) passes a value > 0.

        Returns:
            numpy.ndarray: RGB888 frame array with shape (H, W, 3).

        Raises:
            RuntimeError: If hardware has not been initialized, or if the capture fails.
        """
        if self._camera is None or self._mux is None:
            raise RuntimeError(
                "Hardware not initialized. Call initialize_hardware() first."
            )
        if self._hardware_wedged_message is not None:
            # Already confirmed stuck at the OS level — fail fast instead of piling
            # up more orphaned recovery threads while the operator hasn't restarted yet.
            raise RuntimeError(self._hardware_wedged_message)
        try:
            self._mux.select_channel(channel, settle_delay=settle_delay)

            applied_settings = False
            if isinstance(self._camera, IControllableCamera):
                if exposure_time is not None:
                    self._camera.set_exposure_time(int(exposure_time))
                    applied_settings = True
                if lens_position is not None:
                    self._camera.set_lens_position(float(lens_position))
                    applied_settings = True

                if applied_settings:
                    # Short delay to allow settings to take effect before capture.
                    time.sleep(0.1)

            view_name = f"builder_preview_{channel}"
            frame = self._camera.capture_frame(view_name)

            if applied_settings and isinstance(self._camera, IControllableCamera):
                self._camera.restore_preview_settings()

            self._channel_status[channel] = None
            return frame
        except Exception as exc:
            self._channel_status[channel] = str(exc)
            self._recover_camera_after_failure(channel)
            raise

    def warmup_all_channels(self) -> None:
        """
        Diagnostic-only: capture one throwaway preview frame per camera channel,
        discarding the result (never shown in the UI).

        Failures reuse ``capture_preview_frame()``'s own recovery path and are
        logged, never raised, so one bad channel never blocks warm-up of the rest
        or the Load Sequence call that triggered it.

        Returns:
            None
        """
        for channel in self.get_camera_ports():
            try:
                self.capture_preview_frame(channel, settle_delay=0.1)
            except Exception as exc:
                print(f"[WARN] warmup_all_channels: channel '{channel}' failed: {exc}")

    def get_channel_status(self) -> dict[str, str | None]:
        """
        Return the last known health status per camera channel.

        Refreshed on every ``capture_preview_frame()`` call — never a sticky
        circuit-breaker; a channel simply reflects the outcome of its most
        recent preview attempt.

        Returns:
            dict[str, str | None]: Channel → ``None`` if healthy, or the last
                error message if the most recent attempt failed.
        """
        return dict(self._channel_status)

    def get_hardware_wedged_message(self) -> str | None:
        """
        Return the latched hardware-wedged alarm message, or ``None`` if healthy.

        Set once ``_recover_camera_after_failure()`` fails to bring the camera
        back up at all — never auto-clears, since the underlying libcamera
        resource cannot be released without a full process restart.

        Returns:
            str | None: Operator-facing message, or ``None``.
        """
        return self._hardware_wedged_message

    def _recover_camera_after_failure(self, channel: str | None = None) -> None:
        """
        Best-effort camera + MUX recovery after a preview capture failure.

        Never raises — logs a warning and gives up if recovery itself fails,
        since the caller is already re-raising the original capture error. If
        ``initialize_camera()`` itself cannot bring the camera back up, latches
        ``_hardware_wedged_message`` so further preview attempts fail fast
        instead of retrying a recovery that structurally cannot succeed.
        Mirrors ``SequenceExecutor.recover_hardware()`` but scoped to the
        preview flow (Builder and Inspection on-demand preview).

        Args:
            channel (str | None): The channel that failed, if known. Used to
                bounce the MUX through a different channel before returning,
                since re-selecting the same channel would just resend the
                same I2C value that was already stuck.

        Returns:
            None
        """
        try:
            with self._camera.hw_lock:
                try:
                    self._camera.close_camera()
                except Exception as e:
                    # Not fatal on its own — still attempt initialize_camera() below,
                    # since close_camera() already abandons the instance on timeout.
                    print(f"[WARN] _recover_camera_after_failure: close_camera failed: {e}")
                self._camera.initialize_camera()
            self._mux.reinitialize()
            # Force a real electrical transition on the MUX chip before the next attempt reuses the same channel.
            bounce_channel = next((c for c in self.get_camera_ports() if c != channel), None)
            if bounce_channel is not None:
                self._mux.select_channel(bounce_channel)
        except Exception as e:
            print(f"[WARN] _recover_camera_after_failure: {e}")
            # self._log_power_status("_recover_camera_after_failure")  # diagnostic, disabled (kept for reactivation)
            # self._log_hardware_diagnostics("_recover_camera_after_failure")  # diagnostic, disabled (kept for reactivation)
            self._hardware_wedged_message = (
                "Camera subsystem is stuck and cannot recover automatically. "
                "Use the 'Restart Iris' button (top-right) to restart the service."
            )

    @staticmethod
    def _log_hardware_diagnostics(context: str) -> None:
        """Log dmesg tail and i2cdetect output to diagnose hardware-level CSI/I2C failures."""
        try:
            result = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=5)
            tail = "\n".join(result.stdout.strip().splitlines()[-20:])
            print(f"[INFO] {context}: dmesg tail —\n{tail}")
        except Exception as e:
            print(f"[WARN] {context}: could not read dmesg: {e}")

        try:
            result = subprocess.run(["i2cdetect", "-y", "10"], capture_output=True, text=True, timeout=5)
            print(f"[INFO] {context}: i2cdetect -y 10 —\n{result.stdout.strip()}")
        except Exception as e:
            print(f"[WARN] {context}: could not read i2cdetect: {e}")

    @staticmethod
    def _log_power_status(context: str) -> None:
        """Log ``vcgencmd get_throttled`` to correlate camera failures with under-voltage events."""
        try:
            result = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2)
            throttled = int(result.stdout.strip().split("=")[1], 16)
        except Exception as e:
            print(f"[WARN] {context}: could not read vcgencmd get_throttled: {e}")
            return

        flags = []
        if throttled & 0x1:
            flags.append("under-voltage NOW")
        if throttled & 0x4:
            flags.append("throttled NOW")
        if throttled & 0x10000:
            flags.append("under-voltage occurred since boot")
        if throttled & 0x40000:
            flags.append("throttling occurred since boot")

        if flags:
            print(f"[WARN] {context}: power status 0x{throttled:x} — {', '.join(flags)}")
        else:
            print(f"[INFO] {context}: power status 0x{throttled:x} — nominal")

    def create_calibration_controller(self) -> "GuiCalibrationAdapter":
        """
        Build camera components and return a ``GuiCalibrationAdapter`` for
        on-device PaDiM calibration (sweep + fit).

        The inference engine is intentionally not loaded — this method only
        initializes the hardware needed for feature extraction and capture.

        Returns:
            GuiCalibrationAdapter: Ready to call ``start_sweep()`` or
                ``start_calibration()``.
        """
        self._build_hardware()

        from app.src.adapters.input.GuiCalibrationAdapter import GuiCalibrationAdapter
        from app.src.adapters.output.PaDiMFeatureExtractorAdapter import PaDiMFeatureExtractorAdapter
        from app.src.core.services.CalibrationService import CalibrationService

        extractor = PaDiMFeatureExtractorAdapter()
        service   = CalibrationService(extractor=extractor)
        return GuiCalibrationAdapter(service=service, camera=self._camera)

    def get_calibration_image_dirs(
        self, view_names: list[str]
    ) -> dict[str, dict[str, str]]:
        """
        Build the image directory mapping for ``CalibrationService``.

        Derives paths from ``images_path`` in the sequence JSON.
        The subdirectory structure is::

            {images_path}/train/OK/{view_name}/
            {images_path}/test/OK/{view_name}/
            {images_path}/test/NOK/{view_name}/

        The channel is extracted from each view_name as the last ``_``-separated
        segment (e.g. ``section_view_A`` → channel ``A``).

        Args:
            view_names (list[str]): View names for which to build the mapping.

        Returns:
            dict[str, dict[str, str]]: Mapping of
                ``view_name → {train_ok, test_ok, test_nok}`` as absolute paths.
        """
        base = (
            self._sequence["paths"].get("images_path")
            or self._sequence["paths"].get("train_images_path",
                                           f"./data/images/{self._sequence['part_model']}/")
        )
        return {
            view: {
                "train_ok":  os.path.join(base, "train", "OK",  view),

                "test_ok":   os.path.join(base, "test",  "OK",  view),
                "test_nok":  os.path.join(base, "test",  "NOK", view),
            }
            for view in view_names
        }

    def create_capture_review_service(self) -> "CaptureReviewService":
        """
        Build a ``CaptureReviewService`` for the current sequence's image tree.

        Used by the calibration page's Step 1 review/relabel gallery (Train
        OK / Test OK / Test NOK / Discarded) — see ``/api/review/images`` and
        ``/api/review/relabel`` in ``IrisServer.py``.

        Returns:
            CaptureReviewService: Ready to call ``list_images()`` or
                ``relabel_images()``.
        """
        from app.src.core.services.CaptureReviewService import CaptureReviewService

        images_path = (
            self._sequence["paths"].get("images_path")
            or self._sequence["paths"].get("train_images_path",
                                           f"./data/images/{self._sequence['part_model']}/")
        )
        return CaptureReviewService(images_path=images_path)

    def create_traceability_review_service(self) -> "TraceabilityReviewService":
        """Build a ``TraceabilityReviewService`` for the current sequence.

        Derives paths from the sequence JSON:
        - ``traceability_inference_path``: where JSONL files live.
        - ``inference_images_path``: where per-part inference images are saved.
        - ``images_path``: base for the training/test image tree.

        Returns:
            TraceabilityReviewService: Ready to call ``analyze()`` or
                ``promote_images()``.
        """
        from app.src.core.services.TraceabilityReviewService import TraceabilityReviewService

        paths                   = self._sequence["paths"]
        traceability_path       = paths.get("traceability_inference_path", "")
        inference_images_path   = paths.get("inference_images_path", "")
        images_path             = (
            paths.get("images_path")
            or paths.get("train_images_path",
                         f"./data/images/{self._sequence['part_model']}/")
        )
        return TraceabilityReviewService(
            traceability_path=traceability_path,
            inference_images_path=inference_images_path,
            images_path=images_path,
        )
