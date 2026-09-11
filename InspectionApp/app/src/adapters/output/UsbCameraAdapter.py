import time

import cv2
import numpy as np

from app.src.interfaces.ICamera import ICamera


class UsbCameraAdapter(ICamera):
    """
    Camera adapter for USB cameras using OpenCV ``VideoCapture``.

    Implements the minimal ``ICamera`` contract — no lens or exposure control.
    Suitable for PC development, Jetson, or any platform with a USB/V4L2 camera.

    A single ``VideoCapture`` object is reused for both full-resolution captures
    and low-resolution preview frames.  Resolution is set once at
    ``initialize_camera()``; changing it at runtime is not supported.

    Attributes:
        _capture_resolution (tuple[int, int]): Width × height for ``capture_frame()``.
        _preview_resolution (tuple[int, int]): Width × height for preview JPEG.
        _camera_index (int): ``VideoCapture`` device index (default ``0``).
        _cap (cv2.VideoCapture | None): Active capture object.
    """

    _JPEG_PREVIEW_QUALITY = 65
    _MAX_CAPTURE_RETRIES  = 3
    _RETRY_DELAY_S        = 0.05
    # Number of frames to discard from the OpenCV internal buffer before a
    # capture_frame() call. VideoCapture keeps a FIFO queue that fills
    # continuously; without draining it, capture_frame() returns a stale frame
    # that may be several seconds old. grab() discards frames without decoding.
    _BUFFER_DRAIN_FRAMES  = 5

    def __init__(
        self,
        capture_resolution: tuple[int, int],
        preview_resolution: tuple[int, int],
        camera_index: int = 0,
    ):
        """
        Args:
            capture_resolution (tuple[int, int]): (width, height) for full captures.
            preview_resolution (tuple[int, int]): (width, height) for preview stream.
            camera_index (int): OpenCV device index. Defaults to ``0``.
        """
        self._capture_resolution = capture_resolution
        self._preview_resolution = preview_resolution
        self._camera_index       = camera_index
        self._cap: cv2.VideoCapture | None = None

    # =========================================================================
    # ICamera interface
    # =========================================================================

    def initialize_camera(self) -> None:
        """
        Open the USB camera and configure the capture resolution.

        Note:
            ``cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)`` is a hint to the V4L2 driver.
            The driver may silently round it to the nearest supported sensor mode.
            The actual resolution is read back with ``cap.get()`` after the set and
            stored in ``_capture_resolution`` so that callers always receive frames
            whose dimensions match ``_capture_resolution``.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"UsbCameraAdapter: cannot open camera at index {self._camera_index}."
            )
        w, h = self._capture_resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Request the smallest possible internal buffer so fewer stale frames
        # accumulate between on-demand captures. Not all V4L2 backends honour
        # this; the grab() drain in capture_frame() handles the rest.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read back the resolution that the driver actually accepted.
        # V4L2 rounds the requested value to the nearest supported sensor mode
        # without raising an error. If it differs from the requested value,
        # update _capture_resolution so ROI coordinates and canvas scale factors
        # are always consistent with the real frame size.
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (w, h):
            print(
                f"[WARN] UsbCameraAdapter: requested {w}×{h} but driver accepted "
                f"{actual_w}×{actual_h}. Using actual resolution."
            )
            self._capture_resolution = (actual_w, actual_h)
        print(f"[OK] UsbCameraAdapter: camera {self._camera_index} opened at "
              f"{actual_w}×{actual_h}.")

    def capture_frame(self, view_name: str) -> np.ndarray:
        """
        Capture a full-resolution RGB frame.

        Retries up to ``_MAX_CAPTURE_RETRIES`` times on failure.

        Args:
            view_name (str): Identifier for logging purposes.

        Returns:
            np.ndarray: RGB888 array with shape (H, W, 3).

        Raises:
            RuntimeError: If the camera is not initialized or all retries fail.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(
                "UsbCameraAdapter: camera not initialized. Call initialize_camera() first."
            )
        # Drain the internal OpenCV capture buffer so the returned frame reflects
        # the current scene rather than a queued stale one. grab() skips decoding
        # and is cheap. This is necessary because VideoCapture continuously
        # fills its buffer even when no one is reading.
        for _ in range(self._BUFFER_DRAIN_FRAMES):
            self._cap.grab()
        for attempt in range(self._MAX_CAPTURE_RETRIES):
            ret, frame_bgr = self._cap.read()
            if ret and frame_bgr is not None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            print(f"[WARN] UsbCameraAdapter: capture failed for '{view_name}' "
                  f"(attempt {attempt + 1}/{self._MAX_CAPTURE_RETRIES}).")
            time.sleep(self._RETRY_DELAY_S)
        raise RuntimeError(
            f"UsbCameraAdapter: all {self._MAX_CAPTURE_RETRIES} capture attempts failed "
            f"for view '{view_name}'."
        )

    def get_preview_frame_to_HTML(self) -> bytes:
        """
        Capture a preview frame scaled to preview resolution and return JPEG bytes.

        Returns:
            bytes: JPEG-encoded frame, or empty bytes if the camera is unavailable.
        """
        if self._cap is None or not self._cap.isOpened():
            return b""
        ret, frame_bgr = self._cap.read()
        if not ret or frame_bgr is None:
            return b""
        pw, ph = self._preview_resolution
        preview = cv2.resize(frame_bgr, (pw, ph))
        _, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, self._JPEG_PREVIEW_QUALITY])
        return buf.tobytes()

    def close_camera(self) -> None:
        """Release the ``VideoCapture`` object."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print("[OK] UsbCameraAdapter: camera closed.")

    def release_camera(self) -> None:
        """Force-release the camera (same as ``close_camera`` for USB)."""
        self.close_camera()
