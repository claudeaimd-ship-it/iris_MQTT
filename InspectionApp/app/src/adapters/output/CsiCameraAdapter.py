import gc
import cv2
import time
import threading
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

from app.src.interfaces.IControllableCamera import IControllableCamera


class CsiCameraAdapter(IControllableCamera):
    """
    Adapter for a CSI camera (e.g. Raspberry Pi Camera Module).

    Implements IControllableCamera using the Picamera2 library.
    Configuration parameters are expected to be loaded externally (e.g. from a
    JSON file) and passed at instantiation time.

    Attributes:
        capture_resolution (tuple[int, int]): Resolution used when capturing a full-quality frame (width, height).
        preview_resolution (tuple[int, int]): Resolution used for preview frames (width, height).
        preview_time_exposure (int): Exposure time in microseconds used during preview.
        supports_af_motor (bool): True if the camera sensor has an electronic focus motor
            (e.g. imx708 in RPi Camera Module v3). When False, AfMode and LensPosition
            controls are omitted from Picamera2 initialization and ``set_lens_position()``
            becomes a no-op with a warning. Fixed-focus and CS-mount lens cameras
            (e.g. imx477/HQ Camera) must set this to False.
        picam_instance (Picamera2 | None): Picamera2 camera object. None until initialized.
        _is_initialized (bool): Internal flag indicating whether the camera is ready to capture.
        _is_busy (bool): True while a capture_request is outstanding and unreleased. Test-phase
            flag: currently sticky on failure (no recovery attempted) — see capture_frame().
        hw_lock (threading.RLock): Serializes every Picamera2 call across threads — shared with
            RpiCsiMuxAdapter so a channel switch can never interleave with a capture.
    """

    MAX_RETRIES: int = 3  # Maximum number of initialization attempts before raising an exception.

    def __init__(
        self,
        capture_resolution: tuple[int, int],
        preview_resolution: tuple[int, int],
        preview_time_exposure: int,
        supports_af_motor: bool = True,
        hw_lock: "threading.RLock | None" = None,
    ):
        """
        Args:
            capture_resolution (tuple[int, int]): Resolution used when capturing a full-quality frame (width, height).
            preview_resolution (tuple[int, int]): Resolution used for preview frames (width, height).
            preview_time_exposure (int): Exposure time in microseconds used during preview.
            supports_af_motor (bool): Whether the camera sensor has an electronic focus motor.
                Resolved by AppFactory from ``config/camera_catalog.json`` using the
                ``camera_model`` declared in the sequence JSON. Defaults to True.
            hw_lock (threading.RLock | None): Lock shared with the CSI MUX adapter. A new one is
                created if omitted (e.g. single-camera setups with no MUX).
        """
        self.capture_resolution: tuple[int, int] = capture_resolution
        self.preview_resolution: tuple[int, int] = preview_resolution
        self.preview_time_exposure: int = preview_time_exposure
        self.supports_af_motor: bool = supports_af_motor
        self.picam_instance: Picamera2 | None = None
        self._is_initialized: bool = False
        self._is_busy: bool = False  # True while a capture_request is outstanding (unreleased).
        self.hw_lock: threading.RLock = hw_lock if hw_lock is not None else threading.RLock()

    def initialize_camera(self) -> None:
        """
        Initialize the CSI camera for capturing frames.

        On each attempt, the previous Picamera2 instance is fully destroyed before
        creating a new one to avoid resource conflicts caused by stale handles,
        which is a common source of instability with CSI multi-camera setups.

        Raises:
            RuntimeError: If the camera cannot be initialized after MAX_RETRIES attempts.

        Returns:
            None
        """
        with self.hw_lock:
            for attempt in range(1, self.MAX_RETRIES + 1):
                new_cam = None
                try:
                    self.release_camera()  # Always start from a clean state on every attempt.
                    time.sleep(0.5)  # Give libcamera time to fully release from previous Configured state.

                    new_cam = Picamera2()
                    self.picam_instance = new_cam
                    # Dual-stream: main for full-quality capture, lores for lightweight preview.
                    config = self.picam_instance.create_video_configuration(
                        main={"size": self.capture_resolution, "format": "RGB888"},
                        lores={"size": self.preview_resolution, "format": "YUV420"}
                    )
                    self.picam_instance.configure(config)
                    init_controls: dict = {
                        "AwbMode": controls.AwbModeEnum.Fluorescent,
                    }
                    if self.preview_time_exposure == 0:
                        init_controls["AeEnable"] = True
                    else:
                        init_controls["AeEnable"] = False
                        init_controls["ExposureTime"] = self.preview_time_exposure
                    if self.supports_af_motor:
                        init_controls["AfMode"] = controls.AfModeEnum.Manual
                        init_controls["LensPosition"] = 0.0
                    self.picam_instance.set_controls(init_controls)
                    self.picam_instance.options["quality"] = 95
                    self.picam_instance.start()
                    self._is_initialized = True
                    time.sleep(0.02)  # Allow the sensor to stabilize after starting.
                    print("[OK] CSI camera initialized.")
                    return

                except Exception as e:
                    # If Picamera2() raised before being assigned, clean it up explicitly so
                    # libcamera can transition the camera back to Available state.
                    if new_cam is not None and self.picam_instance is None:
                        try:
                            new_cam.close()
                        except Exception:
                            pass
                    print(f"[ERROR] Failed to initialize camera (attempt {attempt}/{self.MAX_RETRIES}): {e}")
                    if attempt < self.MAX_RETRIES:
                        time.sleep(1)  # libcamera needs ~1-2 s to release from Configured state.

            raise RuntimeError(f"CSI camera could not be initialized after {self.MAX_RETRIES} attempts.")

    def stop_stream(self) -> None:
        """
        Pause the camera stream without closing or releasing the camera.

        Called by RpiCsiMuxAdapter before switching CSI channels via GPIO/I2C.
        Bounded by ``_call_picam_with_timeout`` — an ISP already stalled on the
        previous channel must not block this call (and the shared ``hw_lock``)
        forever; on timeout the instance is force-abandoned so the next
        ``initialize_camera()`` always starts clean.

        Returns:
            None
        """
        with self.hw_lock:
            if self.picam_instance is not None and self._is_initialized:
                try:
                    self._call_picam_with_timeout(self.picam_instance.stop)
                except Exception as e:
                    print(f"[WARN] stop_stream: {e}")
                    self.picam_instance = None
                self._is_initialized = False

    def start_stream(self) -> None:
        """
        Resume the camera stream after a CSI channel switch.

        Bounded by ``_call_picam_with_timeout`` for the same reason as
        ``stop_stream()``. If the instance was force-abandoned by a prior
        ``stop_stream()`` timeout, this is a no-op — the caller's next
        ``capture_frame()``/``initialize_camera()`` call surfaces the problem.

        No settle delay is applied here — callers that need one (e.g. the
        diagnostic warm-up pass) apply it themselves via ``ICsiMux.select_channel()``'s
        ``settle_delay`` parameter.

        Returns:
            None
        """
        with self.hw_lock:
            if self.picam_instance is not None and not self._is_initialized:
                try:
                    self._call_picam_with_timeout(self.picam_instance.start)
                    self._is_initialized = True
                except Exception as e:
                    print(f"[WARN] start_stream: {e}")
                    self.picam_instance = None

    _PICAM_CALL_TIMEOUT_S: float = 1.0  # Seconds before a blocking Picamera2 call is aborted.

    def _call_picam_with_timeout(self, fn, *args):
        """
        Call ``fn(*args)`` in a daemon thread and return the result.

        Raises RuntimeError if the call does not complete within
        ``_PICAM_CALL_TIMEOUT_S`` seconds. Protects against libcamera ISP
        stalls — typically observed after the camera runs continuously for
        several days — where ``capture_request()`` or ``capture_array()``
        block the calling thread indefinitely.

        Args:
            fn: Callable to invoke.
            *args: Positional arguments forwarded to ``fn``.

        Returns:
            Return value of ``fn(*args)``.

        Raises:
            RuntimeError: If the call times out or ``fn`` raises.
        """
        result_holder: list = [None]
        exc_holder:    list = [None]
        done_event = threading.Event()

        def _worker() -> None:
            try:
                result_holder[0] = fn(*args)
            except Exception as exc:
                exc_holder[0] = exc
            finally:
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        if not done_event.wait(timeout=self._PICAM_CALL_TIMEOUT_S):
            raise RuntimeError(
                f"Picamera2 call timed out after {self._PICAM_CALL_TIMEOUT_S:.0f} s — "
                "libcamera ISP may be stalled. Camera will be reinitialized."
            )

        if exc_holder[0] is not None:
            raise exc_holder[0]
        return result_holder[0]

    def capture_frame(self, view_name: str) -> np.ndarray:
        """
        Capture a full-quality frame from the CSI camera.

        Reads from the main (high-resolution) stream of the dual-stream configuration.
        Single attempt only: on failure the instance is force-abandoned (same pattern as
        ``stop_stream()``/``close_camera()``) instead of retried, because a timed-out
        ``_call_picam_with_timeout()`` call leaves an unkillable background thread still
        driving this same Picamera2 instance — retrying against it would race a second
        concurrent native call on a non-thread-safe object. Reinitializing before trying
        again is the caller's responsibility.

        Args:
            view_name (str): Name of the view being captured (e.g. 'front_view_A').

        Raises:
            RuntimeError: If the camera is not initialized, busy, or the capture fails.

        Returns:
            np.ndarray: The captured frame as a NumPy array in RGB888 format.
        """
        with self.hw_lock:
            if not self._is_initialized or self.picam_instance is None:
                raise RuntimeError("Camera is not initialized. Call initialize_camera() first.")
            if self._is_busy:
                raise RuntimeError(
                    f"Camera is busy: a previous capture_request was never released "
                    f"(capture_frame '{view_name}' rejected)."
                )

            try:
                self._is_busy = True
                request = self._call_picam_with_timeout(self.picam_instance.capture_request)
                frame = request.make_array("main")
                request.release()
                self._is_busy = False

                # To avoid color inversion issues with OpenCV, convert from RGB to BGR before returning.
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame_bgr
            except Exception as e:
                print(f"[WARN] capture_frame '{view_name}' failed: {e}")
                # Abandon immediately — do not retry against a possibly-still-running instance.
                self.picam_instance = None
                self._is_initialized = False
                raise RuntimeError(f"capture_frame '{view_name}' failed: {e}")

    def get_preview_frame_to_HTML(self) -> bytes:
        """
        Capture a preview frame and return it as JPEG bytes for HTML streaming.

        Reads from the lores (low-resolution) stream of the dual-stream configuration
        and converts from YUV420 to RGB before encoding as JPEG.

        Raises:
            RuntimeError: If the camera is not initialized or JPEG encoding fails.

        Returns:
            bytes: JPEG-encoded frame ready for multipart HTTP streaming.
        """
        with self.hw_lock:
            if not self._is_initialized or self.picam_instance is None:
                raise RuntimeError("Camera is not initialized. Call initialize_camera() first.")

            yuv = self._call_picam_with_timeout(self.picam_instance.capture_array, "lores")
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            del yuv

            ret, buffer = cv2.imencode('.jpg', rgb)
            if not ret:
                raise RuntimeError("Failed to encode preview frame as JPEG.")
            return buffer.tobytes()

    def set_lens_position(self, new_lens_position: float) -> None:
        """
        Set the lens position of the CSI camera.

        No-op with a warning if the camera model has no electronic focus motor
        (``supports_af_motor=False``). This allows sequence JSON files written for
        autofocus cameras to run unchanged on fixed-focus hardware.

        Args:
            new_lens_position (float): Target lens position (e.g. 0.0–10.0 for RPi Cam v3).

        Raises:
            RuntimeError: If the camera has an AF motor but the control call fails.

        Returns:
            None
        """
        if not self.supports_af_motor:
            print(
                f"[WARN] set_lens_position({new_lens_position}) called but this camera has no "
                "AF motor (supports_af_motor=False). Skipping."
            )
            return
        with self.hw_lock:
            try:
                if self._is_initialized and self.picam_instance is not None:
                    self.picam_instance.stop()  # Stop stream to ensure controls are applied correctly.

                self.picam_instance.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": float(new_lens_position)
                })

                if self._is_initialized and self.picam_instance is not None:
                    self.picam_instance.start()  # Restart stream after applying controls.

                time.sleep(0.1)  # Allow the lens to physically settle before capturing.
            except Exception as e:
                raise RuntimeError(f"Failed to set lens position to {new_lens_position}: {e}")

    def set_exposure_time(self, exposure_time: int) -> None:
        """
        Set the exposure time of the CSI camera.

        Args:
            exposure_time (int): Exposure time in microseconds.
                A value of 0 enables automatic exposure (AeEnable=True).

        Raises:
            RuntimeError: If the exposure time could not be set.

        Returns:
            None
        """
        with self.hw_lock:
            try:
                if self._is_initialized and self.picam_instance is not None:
                    self.picam_instance.stop()  # Stop stream to ensure controls are applied correctly.

                if exposure_time == 0:
                    self.picam_instance.set_controls({"AeEnable": True})
                else:
                    self.picam_instance.set_controls({
                        "AeEnable": False,
                        "ExposureTime": int(exposure_time)
                    })

                if self._is_initialized and self.picam_instance is not None:
                    self.picam_instance.start()  # Restart stream after applying controls.

                time.sleep(0.1)  # Allow the sensor to adjust before capturing.
            except Exception as e:
                raise RuntimeError(f"Failed to set exposure time to {exposure_time}: {e}")

    def restore_preview_settings(self) -> None:
        """
        Restore the camera to preview exposure settings after a capture sweep.

        Called at the end of an inspection cycle so that the next preview
        frame is correctly exposed. Matches the ``camara.picam.stop() →
        SetExposicion(preview) → camara.picam.start()`` sequence used in the
        original Frambuesa implementation.

        Returns:
            None
        """
        with self.hw_lock:
            try:
                if not self._is_initialized or self.picam_instance is None:
                    return
                self.picam_instance.stop()
                if self.preview_time_exposure == 0:
                    self.picam_instance.set_controls({"AeEnable": True})
                else:
                    self.picam_instance.set_controls({
                        "AeEnable": False,
                        "ExposureTime": self.preview_time_exposure,
                    })
                self.picam_instance.start()
                #time.sleep(0.05)
            except Exception as e:
                raise RuntimeError(f"Failed to restore preview settings: {e}")

    def close_camera(self) -> None:
        """
        Stop and close the CSI camera, releasing all resources gracefully.

        Both ``stop()`` and ``close()`` are executed via
        ``_call_picam_with_timeout`` so that a libcamera ISP stall (the same
        condition that triggers the capture timeout) cannot cause
        this method to hang indefinitely.  If either call times out the
        instance is force-abandoned: ``picam_instance`` is set to ``None``
        and ``_is_initialized`` to ``False`` so the next
        ``initialize_camera()`` creates a completely fresh handle.

        Returns:
            None
        """
        with self.hw_lock:
            if self.picam_instance is not None:
                try:
                    self._call_picam_with_timeout(self.picam_instance.stop)
                except Exception as e:
                    self.picam_instance = None
                    self._is_initialized = False
                    raise RuntimeError(f"Failed to stop CSI camera. [{e}]")
                try:
                    self._call_picam_with_timeout(self.picam_instance.close)
                except Exception as e:
                    self.picam_instance = None
                    self._is_initialized = False
                    raise RuntimeError(f"Failed to close CSI camera. [{e}]")
                
                gc.collect()
                time.sleep(0.3)  # Give libcamera time to complete the Configured→Available transition.
                print("[OK] CSI camera closed.")

    def release_camera(self) -> None:
        """
        Force-release all CSI camera resources immediately.

        Used during initialization retries and error recovery, where a
        graceful stop may not be possible.

        Returns:
            None
        """
        with self.hw_lock:
            if self.picam_instance is not None:
                try:
                    self.picam_instance.stop()
                except Exception:
                    pass
                try:
                    self.picam_instance.close()
                except Exception:
                    pass
                self.picam_instance = None
                self._is_initialized = False
                gc.collect()
                time.sleep(0.3)  # Give libcamera time to complete the Configured→Available transition.
            else:
                self.picam_instance = None
                self._is_initialized = False
                gc.collect()
                time.sleep(0.3)  # Give libcamera time to complete the Configured→Available transition.
            
