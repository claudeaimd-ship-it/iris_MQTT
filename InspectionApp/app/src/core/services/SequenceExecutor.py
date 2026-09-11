import threading
import time

import numpy as np

from app.src.core.models.Part import Part, TriggerEvent
from app.src.core.services.InspectionService import InspectionService
from app.src.interfaces.IGpio import IGpio
from app.src.interfaces.ICsiMux import ICsiMux
from app.src.interfaces.ICamera import ICamera
from app.src.interfaces.IControllableCamera import IControllableCamera


class SequenceExecutor:
    """
    Orchestrates a full inspection cycle by executing the steps defined in a
    sequence JSON file.

    Reads the 'steps' list from the sequence, classifies each step by its
    step_number, and dispatches to the appropriate adapter:

    - Normal steps  (step_number 0–999):  GPIO actions and camera captures.
    - Inference steps (step_number ≥ 1001): Trigger InspectionService with all
      captured frames.
    - NOK steps (step_number < 0): GPIO actions executed only when the
      inspection result is NOK.

    The SequenceExecutor does not contain business logic — it is a pure
    coordinator between adapters and the domain service.

    Attributes:
        _gpio (IGpio): GPIO adapter for controlling outputs and reading inputs.
        _mux (ICsiMux): CSI MUX adapter for switching camera channels.
        _camera (ICamera): Camera adapter for capturing frames. If the camera
            also implements ``IControllableCamera``, exposure and lens controls
            are applied; otherwise those steps are silently skipped.
        _inspection_service (InspectionService): Domain service for running inference.
        _steps (list[dict]): Ordered list of steps from the sequence JSON.
        _default_timeout_ms (int): Fallback timeout for wait_for_input if not
            specified in the step.
    """

    def __init__(
        self,
        gpio: IGpio,
        mux: ICsiMux,
        camera: ICamera,
        inspection_service: InspectionService,
        sequence: dict,
        default_timeout_ms: int = 8000,
    ):
        """
        Args:
            gpio (IGpio): GPIO adapter.
            mux (ICsiMux): CSI MUX adapter.
            camera (IControllableCamera): Camera adapter.
            inspection_service (InspectionService): Inference and scoring domain service.
            sequence (dict): Parsed sequence JSON dict. Must contain a 'steps' list
                and a 'part_model' key identifying the type of part being inspected.
            default_timeout_ms (int): Fallback timeout used when a wait_for_input step
                does not specify one. Defaults to 8000 ms.
        """
        self._gpio = gpio
        self._mux = mux
        self._camera = camera
        self._inspection_service = inspection_service
        self._default_timeout_ms = default_timeout_ms
        self._model_id: str = sequence.get("part_model", "unknown")
        self._preprocessing = sequence.get("preprocessing_image_parameters", [])

        # Separate steps by category at construction time to avoid re-filtering on every run.
        all_steps: list[dict] = sequence.get("steps", [])
        self._normal_steps    = sorted([s for s in all_steps if 0 <= s["step_number"] <= 999],  key=lambda s: s["step_number"])
        self._inference_steps = sorted([s for s in all_steps if s["step_number"] >= 1001],      key=lambda s: s["step_number"])
        self._nok_steps       = sorted([s for s in all_steps if s["step_number"] < 0],          key=lambda s: s["step_number"], reverse=True)

        dead_zone = [s["step_number"] for s in all_steps if s["step_number"] == 1000]
        if dead_zone:
            print(f"[WARN] SequenceExecutor: step_number=1000 is never executed (reserved dead zone). "
                  f"Use 0–999 for capture steps or 1001+ for post-inference steps.")

        self._last_captured_frames: dict[str, np.ndarray] = {}
        self._frames_lock = threading.Lock()

        # Used by wait_for_piece_action to exit its polling loop when stop_loop() is called.
        self._stop_event = threading.Event()

        # Live cycle timer for the in-process cycle watchdog (IrisServer._cycle_watchdog).
        # Deliberately separate from part._actual_start_time: this is the only state that
        # survives outside the local `part` object while run() is still executing, so an
        # external thread can read it. None whenever no cycle is actively processing (loop
        # stopped/paused) OR blocked inside ANY indefinite wait_for_input trigger wait —
        # a sequence may chain more than one of these (e.g. "Esperar inicio de ciclo" then
        # "Esperar pieza 1ra pos") and each one pauses the timer again while it blocks; see
        # _execute_gpio_action().
        self._live_cycle_started_at: float | None = None
        self._cycle_timer_lock = threading.Lock()

    # =========================================================================
    # Public API
    # =========================================================================

    def run(self, part_id: str, dry_run: bool = False, forced_scrap: bool = False) -> Part:
        """
        Execute a full inspection cycle for the given part.

        Steps:
        1. Execute normal steps (GPIO + camera captures) in order. Stops early
           if a detect_piece_action determines the piece is absent.
        2. Execute inference steps to evaluate all captured frames (skipped if
           piece is absent.
        3. If the part failed, ``forced_scrap`` is True, and ``dry_run`` is False, execute NOK steps.
        

        Args:
            part_id (str): Unique identifier for this specific physical part instance.
            dry_run (bool): When True, NOK steps (step_number < 0) are skipped.
                The Part result is still recorded normally. Defaults to False.
            forced_scrap (bool): When True, forces the part into NOK status regardless 
                of inference results, causing NOK steps to execute (unless dry_run is True). Defaults to False.

        Returns:
            Part: Completed Part object with inspection results and overall status.
        """
        print(f"[INFO] ── Starting inspection cycle for part: {part_id} ({self._model_id})"
              f"{' [DRY RUN]' if dry_run else ''} ──")
        captured_frames: dict[str, np.ndarray] = {}
        part = Part(part_id, self._model_id)
        part.dry_run = dry_run
        part.forced_scrap = forced_scrap
        part._actual_start_time = time.monotonic()  # For internal timing; not persisted.
        #start_time = time.monotonic()

        try:
            # Step 1: Normal steps — GPIO and capture.
            for step in self._normal_steps:
                self._execute_step(step, part, captured_frames)
                if part.piece_detected is False:
                    print(f"[INFO] Piece not detected — skipping inference steps.")
                    break

            # Step 2: Inference steps — only when piece is present (or not checked).
            if part.piece_detected is not False:
                for step in self._inference_steps:
                    self._execute_inference_step(step, part, captured_frames)

        except TimeoutError as e:
            print(f"[ERROR] Sequence aborted — {e}")
            part.system_error_paused = True

        except Exception as e:
            import traceback
            print(f"[ERROR] Unexpected error during sequence execution: {e}")
            traceback.print_exc()   # Log the full stack trace for debugging.
            part.system_error_paused = True

        # Step 3: NOK steps — conditional GPIO actions.
        if getattr(part, "system_error_paused", False):
            print(f"[WARNING] Cycle aborted due to system error. Skipping NOK steps to prevent PLC/Machine interference.")
        elif not part.overall_status or part.forced_scrap:
            if dry_run:
                print(f"[INFO] Part {part_id} is NOK — NOK steps skipped (dry-run mode).")
            else:
                print(f"[INFO] Part {part_id} is NOK — executing NOK steps.")
                for step in self._nok_steps:
                    self._execute_step(step, part, captured_frames)

        # Cronometer stop and final logging.
        part.time_inspected = time.monotonic() - part._actual_start_time

        # Cycle is done — clear the watchdog-visible timer so idle time until the
        # next trigger is never mistaken for a hung cycle.
        with self._cycle_timer_lock:
            self._live_cycle_started_at = None

        status = ("OK" if part.overall_status else "NOK") + ("_SCRAP" if part.forced_scrap else "") + ("_DR" if part.dry_run else "")
        if getattr(part, "system_error_paused", False):
            status = "ERROR_ABORTED"
        
        print(f"[INFO] ── Inspection complete: {status} ──")

        with self._frames_lock:
            self._last_captured_frames = dict(captured_frames)

        # Save results at the end to ensure all steps have completed and the Part is fully populated.
        self._inspection_service.save_results(part, captured_frames)

        return part

    def interrupt(self) -> None:
        """
        Signal any blocking wait to return immediately.

        Sets both the GPIO interrupt flag (for ``wait_for_input`` steps) and
        the internal ``_stop_event`` (for ``wait_for_piece_action`` polling
        loops). Called by ``stop_loop()`` so the background thread exits within
        the next poll interval rather than waiting up to 15 s for the join.

        Returns:
            None
        """
        self._gpio.interrupt()
        self._stop_event.set()

    def get_live_cycle_age_s(self) -> float | None:
        """
        Seconds elapsed since the in-flight cycle's start trigger was received,
        or ``None`` if no cycle is actively processing right now.

        Used by the in-process cycle watchdog (``IrisServer._cycle_watchdog``)
        to detect a hung cycle. Deliberately NOT based on when ``run()`` was
        called: any ``wait_for_input`` step with no timeout can legitimately
        block for hours between production runs (a sequence may chain more
        than one, e.g. "Esperar inicio de ciclo" then "Esperar pieza 1ra
        pos"), so the timer is paused for the whole duration of each such
        wait and only resumes once that wait's trigger signal actually
        arrives (mirrors ``part._actual_start_time``). This keeps the
        watchdog scoped to the bounded remainder of the cycle (captures,
        bounded waits, inference) instead of counting idle time spent
        waiting to detect a trigger.

        Returns:
            float | None: Seconds since trigger, or None if idle/no active cycle.
        """
        with self._cycle_timer_lock:
            started_at = self._live_cycle_started_at
        if started_at is None:
            return None
        return time.monotonic() - started_at

    def reset_interrupt(self) -> None:
        """
        Clear all interrupt flags before starting a new loop.

        Must be called from ``start_loop()`` so that subsequent blocking steps
        (``wait_for_input``, ``wait_for_piece_action``) work normally on the
        new cycle.

        Returns:
            None
        """
        self._gpio.reset_interrupt()
        self._stop_event.clear()

    def get_last_captured_frames(self) -> dict[str, np.ndarray]:
        """Return a copy of the captured frames from the most recent inspection run.

        Thread-safe — safe to call from an HTTP handler while the inspection loop
        runs in the background.

        Returns:
            dict[str, np.ndarray]: Map of view_name to RGB888 numpy frame.
                Empty if no cycle has completed yet.
        """
        with self._frames_lock:
            return dict(self._last_captured_frames)

    def recover_hardware(self) -> None:
        """
        Recover the camera and CSI MUX in place after a failed cycle.

        Closes and re-initializes the camera, then reinitializes the MUX
        (``ICsiMux.reinitialize()``) so its GPIO selector handles are
        refreshed and the next ``select_channel()`` call re-applies the
        GPIO/I2C configuration instead of skipping it as a no-op.

        A camera-only reset (without also reinitializing the MUX) has been
        observed to be insufficient on multi-camera CSI setups — the MUX
        keeps assuming the previously selected channel is still correctly
        routed and never re-applies it. This mirrors what a full sequence
        reload already does (which recreates both objects from scratch),
        without discarding this executor, its inspection service, or its
        loaded inference models.

        ``close_camera()`` and ``initialize_camera()`` are wrapped in a single
        ``hw_lock`` block — otherwise another thread (e.g. a Builder preview
        request) can acquire the lock in the gap between the two calls and
        run ``start_stream()`` against the just-closed instance, raising
        ``'NoneType' object is not subscriptable`` inside Picamera2.

        Raises:
            Exception: Re-raises any error from the camera or MUX recovery
                so the caller can log it and keep its own retry/backoff logic.

        Returns:
            None
        """
        with self._camera.hw_lock:
            self._camera.close_camera()
            self._camera.initialize_camera()
        self._mux.reinitialize()

    # =========================================================================
    # Step dispatchers
    # =========================================================================

    def _execute_step(self, step: dict, part: Part, captured_frames: dict) -> None:
        """Execute a single normal or NOK step."""
        step_num = step["step_number"]
        description = step.get("description", "")
        print(f"[INFO] Step {step_num}: {description}")

        if "wait_for_piece_action" in step:
            self._execute_wait_for_piece_action(step["wait_for_piece_action"], part)

        if "detect_piece_action" in step:
            self._execute_detect_piece_action(step["detect_piece_action"], part)

        if "gpio_action" in step:
            wait_params = step.get("wait_for_input_parameters", {})
            timeout_ms = wait_params.get("timeout", self._default_timeout_ms)
            expected = wait_params.get("expected_value", 1)
            for action in step["gpio_action"]:
                self._execute_gpio_action(action, timeout_ms, expected, part, step_num)

        if "camera_action" in step:
            for action in step["camera_action"]:
                self._execute_camera_action(action, captured_frames, part)

        delay_ms = step.get("delay_after_step", 0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

    def _execute_inference_step(self, step: dict, part: Part, captured_frames: dict) -> None:
        """Execute a post-capture step (step_number ≥ 1001).

        These steps run after all normal captures are complete. They may contain:
        - ``inference_action``: trigger the full inspection scoring pipeline.
        - ``gpio_action``: send signals after inference results are known (e.g. OK pulse to PLC).

        ``restore_preview_settings()`` is called once at the end of the last
        inference step so the MJPEG stream is correctly exposed for the next cycle.
        """
        step_num   = step["step_number"]
        description = step.get("description", "")
        print(f"[INFO] Step {step_num}: {description}")

        for action in step.get("inference_action", []):
            if action == "execute_full_inspection":
                self._inspection_service.execute_full_inspection_from_frames(part, captured_frames)
            else:
                print(f"[WARN] Unknown inference action: '{action}' — skipped.")

        if "gpio_action" in step:
            wait_params = step.get("wait_for_input_parameters", {})
            timeout_ms = wait_params.get("timeout", self._default_timeout_ms)
            expected = wait_params.get("expected_value", 1)
            for action in step["gpio_action"]:
                self._execute_gpio_action(action, timeout_ms, expected, part, step_num)

        delay_ms = step.get("delay_after_step", 0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

        # Restore preview exposure so the next preview frame is correctly lit.
        # Only applicable for cameras that implement IControllableCamera.
        if isinstance(self._camera, IControllableCamera):
            self._camera.restore_preview_settings()

    # =========================================================================
    # GPIO action executor
    # =========================================================================

    def _execute_gpio_action(self, action: dict, timeout_ms: int, expected: int, part: Part, step_num: int) -> None:
        """Dispatch a single GPIO action dict to the IGpio adapter and record the event."""
        pin    = action["pin_number"]
        op     = action["action"]
        params = action.get("parameters", {})

        if op == "turn_on":
            self._gpio.turn_on(pin)
            part.triggers.append(TriggerEvent(step_num, "OUTPUT", pin, "turn_on", "SENT"))

        elif op == "turn_off":
            self._gpio.turn_off(pin)
            part.triggers.append(TriggerEvent(step_num, "OUTPUT", pin, "turn_off", "SENT"))

        elif op == "send_output":
            value       = params.get("value", 1)
            duration_ms = params.get("duration", 500)
            self._gpio.send_output(pin, value, duration_ms)
            part.triggers.append(TriggerEvent(step_num, "OUTPUT", pin, "send_output", "SENT"))

        elif op == "wait_for_input":
            indefinite = timeout_ms == 0 or timeout_ms is None
            if indefinite:
                # About to block indefinitely waiting for a trigger/piece-presence
                # signal — this can legitimately take minutes or hours between
                # parts (e.g. "Esperar inicio de ciclo" / "Esperar pieza 1ra pos").
                # Pause the watchdog-visible timer while blocked here so it is
                # never mistaken for a hung cycle; it only resumes once this wait
                # actually succeeds, right below.
                with self._cycle_timer_lock:
                    self._live_cycle_started_at = None

            received = self._gpio.wait_for_input(pin, expected, timeout_ms)

            if received and indefinite:
                now = time.monotonic()
                part._actual_start_time = now  # Start timing on the first received signal if not already started.
                # Mirror it into the watchdog-visible timer: the cycle watchdog must only
                # count from here (real trigger received), and gets paused again above
                # if a later step blocks on another indefinite wait_for_input.
                with self._cycle_timer_lock:
                    self._live_cycle_started_at = now

            result_str = "OK" if received else "TIMEOUT"
            part.triggers.append(TriggerEvent(step_num, "INPUT", pin, "wait_for_input", result_str))
            if not received:
                raise TimeoutError(
                    f"Timed out waiting for pin {pin} = {expected} after {timeout_ms} ms."
                )

        else:
            print(f"[WARN] Unknown GPIO action: '{op}' on pin {pin} — skipped.")

    # =========================================================================
    # Camera action executor
    # =========================================================================

    def _execute_camera_action(self, action: dict, captured_frames: dict, part: Part) -> None:
        """
        Dispatch a single camera action dict.

        Selects the MUX channel, applies per-capture parameters (exposure,
        lens position), captures the frame, and stores it in captured_frames
        keyed by a unique view name built from prefix_view + camera_port.
        """
        channel    = action["camera_port"]          # e.g. 'A'
        prefix     = action.get("prefix_view", "")  # e.g. 'front_view_section_1'
        view_name  = f"{prefix}_{channel}" if prefix else channel  # e.g. 'front_view_section_1_A'

        try:
            # Switch MUX to the requested channel (stop → GPIO+I2C → start).
            self._mux.select_channel(channel)

            # Find preprocessing parameters for this view or camera port, if defined. Defaults to empty dict.
            pipeline = []
            for entry in self._preprocessing:
                entry_view = str(entry.get("view", ""))
                entry_port = str(entry.get("camera_port", ""))
                if entry_view == prefix and entry_port == channel:
                    pipeline = entry.get("pipeline", [])
                    break

            # Fallback: if no view-specific parameters, look for camera_port match (legacy support).
            if not pipeline:
                for entry in self._preprocessing:
                    if str(entry.get("camera_port")) == channel and not entry.get("view"):
                        pipeline = entry.get("pipeline", [])
                        break

            # Extract parameters.
            exposure = None
            lens_position = None
            for tool in pipeline:
                if tool["tool"] == "set_time_exposure":
                    exposure = tool["parameters"].get("exposure_time")
                elif tool["tool"] == "set_lens_position":
                    lens_position = tool["parameters"].get("lens_position")

            # Apply per-capture settings when provided (only for controllable cameras).
            if isinstance(self._camera, IControllableCamera):
                if exposure is not None:
                    self._camera.set_exposure_time(int(exposure))
                if lens_position is not None:
                    self._camera.set_lens_position(float(lens_position))

            frame = self._camera.capture_frame(view_name)
        except Exception as exc:
            part.failed_channel = channel
            part.failed_channel_error = str(exc)
            raise

        captured_frames[view_name] = frame
        print(f"[OK] Frame captured: {view_name}")

    # =========================================================================
    # Detect piece action executor
    # =========================================================================

    def _execute_detect_piece_action(self, action: dict, part: Part) -> None:
        """
        Capture a frame and determine if a piece is present based on mean ROI brightness.

        The part is black, so a present piece lowers the mean grayscale brightness
        of the ROI below ``darkness_threshold``.

        Args:
            action (dict): detect_piece_action dict with keys ``camera_port``,
                ``roi`` (x, y, w, h), and optional ``darkness_threshold`` (default 80).
            part (Part): Part being inspected. ``piece_detected`` is set in-place.
        """
        camera_port       = action["camera_port"]
        roi               = action["roi"]
        darkness_threshold = action.get("darkness_threshold", 80)

        try:
            self._mux.select_channel(camera_port)
            frame = self._camera.capture_frame(f"detect_piece_{camera_port}")
        except Exception as exc:
            part.failed_channel = camera_port
            part.failed_channel_error = str(exc)
            raise

        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        roi_crop = frame[y:y + h, x:x + w]

        # RGB → grayscale using ITU-R 601 luminance weights (no cv2 import needed).
        gray = np.dot(roi_crop[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
        mean_brightness = float(gray.mean())

        if mean_brightness < darkness_threshold:
            part.piece_detected = True
            print(f"[INFO] Piece detected (brightness={mean_brightness:.1f} < threshold={darkness_threshold}).")
        else:
            part.piece_detected = False
            print(f"[INFO] No piece detected (brightness={mean_brightness:.1f} >= threshold={darkness_threshold}).")

    # =========================================================================
    # Wait for piece action executor (visual trigger)
    # =========================================================================

    def _execute_wait_for_piece_action(self, action: dict, part: Part) -> None:
        """
        Block until a piece is detected in a ROI by comparing against a reference
        background frame captured at the start of this step.

        Captures the first frame as the reference (empty fixture), then polls the
        camera at ``poll_interval_ms`` intervals. On each poll the mean absolute
        grayscale difference between the live ROI and the reference ROI is computed.
        When it reaches ``pixel_diff_threshold`` the piece is considered present and
        the method returns.

        Raises ``TimeoutError`` if:
        - ``timeout_ms > 0`` and the piece is not detected within that time.
        - ``interrupt()`` is called externally (e.g. by ``stop_loop()``).

        The ``TimeoutError`` is caught by ``run()`` which sets
        ``part.system_error_paused = True`` — identical behaviour to an
        interrupted ``wait_for_input`` step.

        Args:
            action (dict): ``wait_for_piece_action`` dict with the following keys:

                - ``camera_port`` (str): Channel to read from (e.g. ``"A"``).
                - ``roi`` (dict): Region of interest with keys ``x``, ``y``, ``w``, ``h``
                  in capture-resolution pixel space.
                - ``pixel_diff_threshold`` (float): Mean absolute grayscale difference
                  (0–255) that triggers piece detection. Default: ``15``.
                - ``poll_interval_ms`` (int): Milliseconds between camera polls.
                  Default: ``100``.
                - ``timeout_ms`` (int): Maximum wait time in milliseconds. ``0`` means
                  wait indefinitely. Default: ``0``.
                - ``stabilization_ms`` (int): Milliseconds to wait after selecting the
                  MUX channel before capturing the reference frame. Useful to let the
                  camera sensor stabilise after a channel switch. Default: ``0``.

            part (Part): Part being inspected. ``piece_detected`` and
                ``_actual_start_time`` are updated in-place on detection.
        """
        camera_port          = action["camera_port"]
        roi                  = action["roi"]
        pixel_diff_threshold = float(action.get("pixel_diff_threshold", 15))
        poll_interval_ms     = int(action.get("poll_interval_ms", 100))
        timeout_ms           = int(action.get("timeout_ms", 0))
        stabilization_ms     = int(action.get("stabilization_ms", 0))

        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

        try:
            self._mux.select_channel(camera_port)

            if stabilization_ms > 0:
                time.sleep(stabilization_ms / 1000)

            # Capture the reference frame (empty fixture / background).
            ref_frame = self._camera.capture_frame("wait_for_piece_ref")
        except Exception as exc:
            part.failed_channel = camera_port
            part.failed_channel_error = str(exc)
            raise
        ref_roi_gray = np.dot(
            ref_frame[y:y + h, x:x + w][..., :3].astype(np.float32),
            [0.299, 0.587, 0.114],
        )
        print(
            f"[INFO] wait_for_piece: reference captured — port={camera_port}, "
            f"ROI=({x},{y},{w},{h}), threshold={pixel_diff_threshold}, "
            f"poll={poll_interval_ms} ms, timeout={timeout_ms} ms."
        )

        start_time = time.monotonic()

        while True:
            if self._stop_event.is_set():
                raise TimeoutError("wait_for_piece_action interrupted by stop signal.")

            time.sleep(poll_interval_ms / 1000)

            if timeout_ms > 0:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                if elapsed_ms >= timeout_ms:
                    raise TimeoutError(
                        f"wait_for_piece_action: no piece detected after {timeout_ms} ms."
                    )

            try:
                frame = self._camera.capture_frame("wait_for_piece_poll")
            except Exception as exc:
                part.failed_channel = camera_port
                part.failed_channel_error = str(exc)
                raise
            roi_gray = np.dot(
                frame[y:y + h, x:x + w][..., :3].astype(np.float32),
                [0.299, 0.587, 0.114],
            )
            mean_diff = float(np.abs(roi_gray - ref_roi_gray).mean())

            if mean_diff >= pixel_diff_threshold:
                part._actual_start_time = time.monotonic()
                part.piece_detected = True
                print(
                    f"[INFO] wait_for_piece: piece detected "
                    f"(mean_diff={mean_diff:.2f} >= threshold={pixel_diff_threshold})."
                )
                return
