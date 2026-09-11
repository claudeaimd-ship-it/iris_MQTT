from abc import ABC, abstractmethod
import numpy as np

class ICamera(ABC):
    """
    Abstract base interface for all camera types.

    Defines the minimal contract that any camera adapter must fulfill,
    regardless of connection type (CSI, USB, IP, etc.).
    For cameras that support hardware control (lens, exposure), see IControllableCamera.
    """

    @abstractmethod
    def initialize_camera(self) -> None:
        """
        Initialize the camera and prepare it for capturing frames.

        Returns:
            None
        """
        pass

    @abstractmethod
    def capture_frame(self, view_name: str) -> np.ndarray:
        """
        Capture a full-quality frame from the camera.

        Args:
            view_name (str): Name of the view being captured (e.g. 'front_view_A').

        Returns:
            np.ndarray: The captured frame as a NumPy array.
        """
        pass

    @abstractmethod
    def get_preview_frame_to_HTML(self) -> bytes:
        """
        Capture a preview frame and return it as JPEG bytes for HTML streaming.

        Returns:
            bytes: JPEG-encoded frame ready for multipart HTTP streaming.
        """
        pass

    @abstractmethod
    def close_camera(self) -> None:
        """
        Stop and close the camera, releasing all resources gracefully.

        Returns:
            None
        """
        pass

    @abstractmethod
    def release_camera(self) -> None:
        """
        Force-release all camera resources immediately.

        Intended for error recovery when a graceful close is not possible.

        Returns:
            None
        """
        pass
