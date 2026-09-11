import os
import threading

import cv2
import numpy as np
import time
from datetime import datetime

from app.src.interfaces.IControllableCamera import IControllableCamera
from app.src.interfaces.ICamera import ICamera
from app.src.interfaces.ICsiMux import ICsiMux
from app.src.interfaces.IGpio import IGpio


class SampleCaptureService:
    """
    Application service for training data capture.

    Coordinates the hardware (camera, MUX, GPIO) needed to execute a single
    labeled capture cycle: wait for a GPIO trigger, turn on spotlights, sweep
    all camera channels, save frames to disk, and restore the preview channel.

    All hardware access goes through interfaces so the service is
    adapter-agnostic. Swapping the MUX for a software-only stub (e.g. for
    single Ethernet cameras) requires no changes in this class.

    Saved directory structure::

        ok      → {images_path}/train/OK/{view_name}/{YYYYMMDD_HHMMSS_ffffff}.jpg
        test_ok → {images_path}/test/OK/{view_name}/{YYYYMMDD_HHMMSS_ffffff}.jpg
        nok     → {images_path}/test/NOK/{view_name}/{YYYYMMDD_HHMMSS_ffffff}.jpg

    This mirrors the PaDiM calibration dataset layout so both the Samples
    capture mode and the Calibration capture mode write to the same
    directories, keeping a single image tree per product model.

    Schedule Timed Captures (design confirmed 2026-07-31): an optional,
    in-memory-only scheduler (never persisted to disk — lost on restart,
    same as ``dry_run``) that automates periodic dataset collection while
    Samples mode keeps running continuously on real production triggers.
    When enabled, every ``_schedule_interval_s`` seconds a capture window
    opens during which the first ``_schedule_images_per_window`` triggered
    cycles are persisted to the schedule's configured label (``'ok'`` or
    ``'test_ok'`` only — Test NOK is intentionally excluded, since NOK
    occurrences are expected to be manually reviewed rather than
    auto-collected); cycles outside the window still run (GPIO handshake,
    camera capture) but are not written to the labeled dataset, only to the
    always-on ``latest/`` preview. The schedule auto-pauses (stops
    persisting, but stays armed) once ``_schedule_target_images`` total
    cycles have been persisted. See ``next_cycle_plan()`` /
    ``schedule_record_persisted()`` / ``enable_schedule()``.

    Attributes:
        _camera (ICamera): Camera adapter for frame capture.
        _mux (ICsiMux): CSI MUX adapter for channel switching.
        _gpio (IGpio): GPIO adapter for spotlights and trigger input.
        _spotlight_pins (list[int]): GPIO pins toggled for illumination.
        _camera_channels (list[str]): Channel names swept on each cycle.
        _trigger_pin (int): GPIO input pin to wait for before each cycle.
        _trigger_timeout_ms (int): Maximum wait time for the trigger in ms.
        _images_path (str): Root directory for all captured images
            (``data/images/{part_model}/``).
        _preview_channel (str | None): MUX channel restored between cycles
            so the preview stream stays on a stable view.
    """

    # Maps label names to their subdirectory under ``_images_path``.
    _LABEL_SUBDIRS: dict[str, str] = {
        "ok":      os.path.join("train", "OK"),
        "test_ok": os.path.join("test",  "OK"),
        "nok":     os.path.join("test",  "NOK"),
    }

    def __init__(
        self,
        camera: ICamera,
        mux: ICsiMux,
        gpio: IGpio,
        spotlight_pins: list[int],
        sequence_steps: list[dict],
        trigger_pin: int,
        trigger_timeout_ms: int,
        images_path: str,
        preprocessing_params: list[dict] = None,
    ):
        """
        Args:
            camera (ICamera): Camera adapter.
            mux (ICsiMux): CSI MUX adapter.
            gpio (IGpio): GPIO adapter.
            spotlight_pins (list[int]): GPIO pin numbers turned on before each
                capture sweep and off afterwards.
            sequence_steps (list[str]): Ordered list of steps to execute during
                each capture cycle (e.g. ``['A', 'B']``).
            trigger_pin (int): GPIO input pin number for the PLC/robot trigger.
            trigger_timeout_ms (int): Maximum time in milliseconds to wait for
                the trigger signal.
            images_path (str): Root directory for all captured images
                (e.g. ``./data/images/my_part/``). Subdirectories are created
                automatically on first save.
        """
        self._camera             = camera
        self._mux                = mux
        self._gpio               = gpio
        self._spotlight_pins     = spotlight_pins
        self._sequence_steps     = sequence_steps
        self._trigger_pin        = trigger_pin
        self._trigger_timeout_ms = trigger_timeout_ms
        self._images_path        = images_path
        self._preprocessing_params   = preprocessing_params or []

        self._views: list[tuple[str,str]] = []
        for step in sequence_steps:
            # Ignore NOK and Inference steps
            if 0 <= step.get("step_number", 0) < 999:
                if "camera_action" in step:
                    for action in step["camera_action"]:
                        ch = action.get("camera_port")
                        prefix = action.get("prefix_view", f"view{ch}")
                        view_name = f"{prefix}_{ch}"
                        if ch:
                            self._views.append((ch, view_name))

        self._preview_channel: str | None = self._views[0][0] if self._views else None

        # Used by _wait_for_piece() to exit its polling loop when stop_loop() is called.
        self._stop_event = threading.Event()

        # Schedule Timed Captures — in-memory only, never persisted to disk.
        self._schedule_enabled: bool = False
        self._schedule_images_per_window: int = 20
        self._schedule_interval_s: float = 90 * 60
        self._schedule_target_images: int = 300
        self._schedule_label: str = "ok"
        self._schedule_window_start: float | None = None
        self._schedule_window_persisted: int = 0
        self._schedule_total_persisted: int = 0

    # =========================================================================
    # Public API
    # =========================================================================

    def interrupt(self) -> None:
        """
        Signal any blocking wait to return immediately.

        Sets both the GPIO interrupt flag (for ``wait_for_input`` steps) and
        the internal ``_stop_event`` (for ``wait_for_piece_action`` polling
        loops). Called by ``stop_loop()`` so the background thread exits within
        the next poll interval.

        Returns:
            None
        """
        self._gpio.interrupt()
        self._stop_event.set()

    def reset_interrupt(self) -> None:
        """
        Clear all interrupt flags before starting a new capture loop.

        Must be called from ``start_loop()`` so that subsequent blocking steps
        (``wait_for_input``, ``wait_for_piece_action``) work normally.

        Returns:
            None
        """
        self._gpio.reset_interrupt()
        self._stop_event.clear()

    def wait_for_trigger(self) -> bool:
        """
        No-op — the actual trigger wait is handled by the first
        ``wait_for_input`` gpio_action step inside ``run_capture_cycle()``.

        Returns:
            bool: Always True.
        """
        return True

    def get_views(self) -> list[tuple[str, str]]:
        """
        Return the ordered list of (channel, view_name) pairs derived from the
        sequence steps at construction time.

        Used by ``GuiSamplesAdapter`` to resolve channel names to view_names
        when writing ``latest/`` thumbnails.

        Returns:
            list[tuple[str, str]]: Each tuple is ``(camera_port, view_name)``,
                e.g. ``[("A", "section_1_A"), ("B", "section_1_B")]``.
        """
        return list(self._views)

    def run_capture_cycle(self, label: str, persist: bool = True) -> dict[str, str]:
        """
        Turn on spotlights, capture one frame per channel, save to disk, and
        restore the preview channel.

        Args:
            label (str): Capture label — ``'ok'``, ``'test_ok'``, or ``'nok'``. Determines the
                destination subdirectory (see ``_LABEL_SUBDIRS``).
            persist (bool): When ``False``, the cycle still runs in full
                (GPIO handshake, camera capture, ``latest/`` preview update)
                but the labeled dataset copy is NOT written to disk — used by
                Schedule Timed Captures to skip persistence between capture
                windows without skipping the physical cycle itself.

        Returns:
            dict[str, str]: Mapping of channel name to saved absolute file
                path, for channels that were persisted (empty when
                ``persist`` is ``False``).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        saved: dict[str, str] = {}

        # for pin in self._spotlight_pins:
        #     self._gpio.turn_on(pin)

        try:
            for step in sorted(self._sequence_steps, key=lambda s: s.get("step_number", 0)):
                step_number = step.get("step_number", 0)
                if step_number < 0 or step_number >= 1000:
                    continue  # Skip non-capture steps (e.g. NOK, Inference)

                print(f"[INFO] SampleCaptureService: Step {step_number}: {step.get('description', '')}")

                # 0. Visual trigger — block until piece detected
                if "wait_for_piece_action" in step:
                    self._wait_for_piece(step["wait_for_piece_action"])

                # 1. Actions GPIO (Waits and signals to PLC)
                if "gpio_action" in step:
                    timeout_ms = step.get("wait_for_input_parameters", {}).get("timeout", 8000)
                    expected = step.get("wait_for_input_parameters", {}).get("expected_value", 1)
                    
                    for action in step["gpio_action"]:
                        pin = action["pin_number"]
                        op = action["action"]
                        params = action.get("parameters", {})

                        if op == "turn_on":
                            self._gpio.turn_on(pin)
                        elif op == "turn_off":
                            self._gpio.turn_off(pin)
                        elif op == "send_output":
                            value = params.get("value", 1)
                            duration_ms = params.get("duration", 500)
                            self._gpio.send_output(pin, value, duration_ms)
                        elif op == "wait_for_input":
                            # Execute the wait in the same way as SequenceExecutor to maintain PLC handshake integrity.
                            received = self._gpio.wait_for_input(pin, expected, timeout_ms)
                            if not received:
                                raise TimeoutError(f"Timed out waiting for pin {pin} = {expected} after {timeout_ms} ms.")

                # 2. Camera Actions (Capture and save)
                if "camera_action" in step:
                    for action in step["camera_action"]:
                        channel = action.get("camera_port")
                        if not channel: continue
                            
                        prefix = action.get("prefix_view", "")
                        view_name = f"{prefix}_{channel}" if prefix else channel

                        self._mux.select_channel(channel)
                        
                        # Search for light and lens parameters
                        pipeline = []
                        for entry in self._preprocessing_params:
                            entry_view = str(entry.get("view", ""))
                            entry_port = str(entry.get("camera_port", ""))
                            if entry_view == prefix and entry_port == channel:
                                pipeline = entry.get("pipeline", [])
                                break
                        if not pipeline:
                            for entry in self._preprocessing_params:
                                if str(entry.get("camera_port")) == channel and not entry.get("view"):
                                    pipeline = entry.get("pipeline", [])
                                    break

                        exposure = None
                        lens_position = None
                        for tool in pipeline:
                            if tool.get("tool") == "set_time_exposure":
                                exposure = tool.get("parameters", {}).get("exposure_time")
                            elif tool.get("tool") == "set_lens_position":
                                lens_position = tool.get("parameters", {}).get("lens_position")

                        if isinstance(self._camera, IControllableCamera):
                            if exposure is not None:
                                self._camera.set_exposure_time(int(exposure))
                            if lens_position is not None:
                                self._camera.set_lens_position(float(lens_position))

                        frame = self._camera.capture_frame(view_name)
                        # latest/ preview is a single overwritten file per view (not a
                        # growing dataset) — always written regardless of `persist`,
                        # same principle as InspectionService's NOK-only persistence.
                        self._save_latest_preview(frame, view_name)
                        if persist:
                            path = self._save_frame(frame, label, view_name, timestamp)
                            saved[channel] = path
                            print(f"[OK] SampleCaptureService: saved {label}/{channel} → {path}")

                # 3. Delays
                delay_ms = step.get("delay_after_step", 0)
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000)

        finally:
            # Safety shutdown in case the sequence is aborted halfway
            for pin in self._spotlight_pins:
                self._gpio.turn_off(pin)
            if self._preview_channel is not None:
                self._mux.select_channel(self._preview_channel)

        return saved

    def select_preview_channel(self, channel: str) -> None:
        """
        Switch the MUX to the given channel and record it as the preview channel.

        Called during initialization and after camera recovery to ensure the
        preview stream shows the correct view between capture cycles.

        Args:
            channel (str): Channel name to switch to (e.g. ``'A'``).
        """
        self._mux.select_channel(channel)
        self._preview_channel = channel

    def restore_preview(self) -> None:
        """
        Switch the MUX back to the stored preview channel.

        Convenience wrapper used during initialization and crash recovery so the
        caller does not need to track which channel is the preview channel.
        No-op if no channels are configured.
        """
        if self._preview_channel is not None:
            self._mux.select_channel(self._preview_channel)

    def count_samples(self, label: str) -> dict[str, int]:
        """
        Count saved ``.jpg`` images per channel for a given label.

        Args:
            label (str): Label name — ``'ok'``, ``'test_ok'``, or ``'nok'``.

        Returns:
            dict[str, int]: Mapping of channel name to number of saved images.
        """
        subdir = self._LABEL_SUBDIRS.get(label, os.path.join(label))
        counts: dict[str, int] = {}
        for _, view_name in self._views:
            folder = os.path.join(self._images_path, subdir, view_name)
            if os.path.isdir(folder):
                counts[view_name] = len(
                    [f for f in os.listdir(folder) if f.endswith(".jpg")]
                )
            else:
                counts[view_name] = 0
        return counts

    def count_all_labels(self) -> dict[str, dict[str, int]]:
        """
        Count samples for all known labels.

        Returns:
            dict[str, dict[str, int]]: Mapping label → channel → count.
        """
        return {label: self.count_samples(label) for label in self._LABEL_SUBDIRS}

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _wait_for_piece(self, action: dict) -> None:
        """
        Block until a piece is detected in a ROI by comparing pixel differences
        against a reference frame captured at the start of this step.

        Identical logic to ``SequenceExecutor._execute_wait_for_piece_action()``
        but without a ``Part`` object — just blocks and returns on detection.
        Raises ``TimeoutError`` on interrupt or timeout so ``run_capture_cycle``
        propagates it up to ``GuiSamplesAdapter``, which handles it with crash
        recovery identical to ``GuiInferenceAdapter``.

        Args:
            action (dict): ``wait_for_piece_action`` dict. See
                ``SequenceExecutor._execute_wait_for_piece_action`` for the full
                key reference.
        """
        camera_port          = action["camera_port"]
        roi                  = action["roi"]
        pixel_diff_threshold = float(action.get("pixel_diff_threshold", 15))
        poll_interval_ms     = int(action.get("poll_interval_ms", 100))
        timeout_ms           = int(action.get("timeout_ms", 0))
        stabilization_ms     = int(action.get("stabilization_ms", 0))

        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

        self._mux.select_channel(camera_port)

        if stabilization_ms > 0:
            time.sleep(stabilization_ms / 1000)

        ref_frame = self._camera.capture_frame("wait_for_piece_ref")
        ref_roi_gray = np.dot(
            ref_frame[y:y + h, x:x + w][..., :3].astype(np.float32),
            [0.299, 0.587, 0.114],
        )
        print(
            f"[INFO] SampleCaptureService wait_for_piece: reference captured — "
            f"port={camera_port}, ROI=({x},{y},{w},{h}), "
            f"threshold={pixel_diff_threshold}, poll={poll_interval_ms} ms."
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

            frame = self._camera.capture_frame("wait_for_piece_poll")
            roi_gray = np.dot(
                frame[y:y + h, x:x + w][..., :3].astype(np.float32),
                [0.299, 0.587, 0.114],
            )
            mean_diff = float(np.abs(roi_gray - ref_roi_gray).mean())

            if mean_diff >= pixel_diff_threshold:
                print(
                    f"[INFO] SampleCaptureService wait_for_piece: piece detected "
                    f"(mean_diff={mean_diff:.2f} >= threshold={pixel_diff_threshold})."
                )
                return

    def _save_frame(
        self,
        frame: np.ndarray,
        label: str,
        view_name: str,
        timestamp: str,
    ) -> str:
        """
        Save a single RGB frame as a JPEG under the appropriate subdirectory.

        Args:
            frame (np.ndarray): RGB image array captured from the camera.
            label (str): Label name (``'ok'``, ``'test_ok'``, or ``'nok'``).
            view_name (str): View name derived from the channel configuration (e.g. ``'viewA_0'``).
            timestamp (str): Timestamp string used as the filename stem.

        Returns:
            str: Absolute path of the saved file.
        """
        subdir    = self._LABEL_SUBDIRS.get(label, label)
        folder    = os.path.join(self._images_path, subdir, view_name)
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"{timestamp}.jpg")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, bgr)
        return os.path.abspath(file_path)

    def _save_latest_preview(self, frame: np.ndarray, view_name: str) -> None:
        """
        Overwrite ``{images_path}/latest/{view_name}.jpg`` with the frame just
        captured, for the Inspection page's "Last Captures" preview.

        Always called from ``run_capture_cycle()`` regardless of the
        ``persist`` flag \u2014 this is a single overwritten file per view (not a
        growing dataset), mirroring ``GuiInferenceAdapter``'s equivalent
        mechanism, so the live preview keeps working even when Schedule Timed
        Captures is skipping labeled-dataset persistence between windows.

        Args:
            frame (np.ndarray): RGB image array captured from the camera.
            view_name (str): View name for the destination filename.
        """
        latest_dir = os.path.join(self._images_path, "latest")
        os.makedirs(latest_dir, exist_ok=True)
        dst_path = os.path.join(latest_dir, f"{view_name}.jpg")
        try:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            if w > 800:
                scale = 800 / w
                bgr = cv2.resize(bgr, (800, int(h * scale)), interpolation=cv2.INTER_AREA)
            _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            tmp = dst_path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(enc.tobytes())
            os.replace(tmp, dst_path)
        except Exception as err:
            print(f"[WARN] SampleCaptureService: could not write latest preview for {view_name}: {err}")

    # =========================================================================
    # Schedule Timed Captures
    # =========================================================================

    def enable_schedule(
        self, images_per_window: int, interval_s: float, target_images: int, label: str
    ) -> None:
        """
        Turn on Schedule Timed Captures with a fresh configuration.

        In-memory only \u2014 never persisted to disk, matching ``dry_run``'s
        behavior; lost on server restart, must be re-armed by the operator.

        Args:
            images_per_window (int): Number of production cycles to persist
                per capture window.
            interval_s (float): Seconds between the start of one window and
                the next.
            target_images (int): Total cycles to persist before the schedule
                auto-pauses (stays enabled but stops persisting).
            label (str): Destination label \u2014 must be ``'ok'`` or
                ``'test_ok'`` (snapshotted from the Samples mode label
                selected at the time the schedule is enabled; Test NOK is not
                a valid schedule destination).
        """
        self._schedule_images_per_window = images_per_window
        self._schedule_interval_s        = interval_s
        self._schedule_target_images     = target_images
        self._schedule_label             = label
        self._schedule_enabled           = True
        self._schedule_window_start      = None  # forces a fresh window on the next tick
        self._schedule_window_persisted  = 0
        self._schedule_total_persisted   = 0

    def disable_schedule(self) -> None:
        """Turn off Schedule Timed Captures. Manual capture behavior resumes immediately."""
        self._schedule_enabled = False

    def is_schedule_enabled(self) -> bool:
        """True if Schedule Timed Captures is currently armed."""
        return self._schedule_enabled

    def get_schedule_status(self) -> dict:
        """
        Return the current Schedule Timed Captures state for status polling.

        Returns:
            dict: ``enabled``, ``images_per_window``, ``interval_s``,
                ``target_images``, ``label``, ``total_persisted``,
                ``window_persisted``, ``target_reached``.
        """
        return {
            "enabled":            self._schedule_enabled,
            "images_per_window":  self._schedule_images_per_window,
            "interval_s":         self._schedule_interval_s,
            "target_images":      self._schedule_target_images,
            "label":              self._schedule_label,
            "total_persisted":    self._schedule_total_persisted,
            "window_persisted":   self._schedule_window_persisted,
            "target_reached":     self._schedule_total_persisted >= self._schedule_target_images,
        }

    def next_cycle_plan(self, manual_label: str) -> tuple[bool, str]:
        """
        Decide whether the upcoming triggered cycle should persist to the
        labeled dataset, and which label to use.

        When the schedule is disabled, always persists using
        ``manual_label`` \u2014 today's default behavior, unchanged. When
        enabled, opens a new capture window every ``_schedule_interval_s``
        seconds and persists only the first ``_schedule_images_per_window``
        cycles within it, using the schedule's own snapshotted label; stops
        persisting (but stays armed) once ``_schedule_target_images`` total
        cycles have been persisted.

        Args:
            manual_label (str): The operator's currently selected Samples
                mode label \u2014 used verbatim when the schedule is disabled.

        Returns:
            tuple[bool, str]: ``(should_persist, label_to_use)``.
        """
        if not self._schedule_enabled:
            return True, manual_label

        if self._schedule_total_persisted >= self._schedule_target_images:
            return False, self._schedule_label

        now = time.monotonic()
        if (
            self._schedule_window_start is None
            or (now - self._schedule_window_start) >= self._schedule_interval_s
        ):
            self._schedule_window_start     = now
            self._schedule_window_persisted = 0

        if self._schedule_window_persisted < self._schedule_images_per_window:
            return True, self._schedule_label

        return False, self._schedule_label

    def schedule_record_persisted(self) -> None:
        """
        Record that a cycle was just persisted under the active schedule.

        No-op when the schedule is disabled \u2014 safe to call unconditionally
        after any persisted cycle.
        """
        if self._schedule_enabled:
            self._schedule_window_persisted += 1
            self._schedule_total_persisted  += 1
