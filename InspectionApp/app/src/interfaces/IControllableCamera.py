from abc import abstractmethod
from app.src.interfaces.ICamera import ICamera


class IControllableCamera(ICamera):
    """
    Extended camera interface for cameras that support hardware control.

    Inherits the base ICamera contract and adds methods for controlling
    lens position and exposure time. Intended for cameras such as CSI
    modules (e.g. Raspberry Pi Camera Module) where these parameters
    can be set programmatically.

    Cameras that do not support these controls (e.g. USB cameras)
    should implement ICamera directly instead.
    """

    @abstractmethod
    def set_lens_position(self, new_lens_position: float) -> None:
        """
        Set the lens position of the camera.

        The implementation should apply the change gradually to avoid
        mechanical stress and focus instability.

        Args:
            new_lens_position (float): Target lens position. Valid range
                depends on the camera module (e.g. 0.0–10.0 for RPi Cam v3).

        Returns:
            None
        """
        pass

    @abstractmethod
    def set_exposure_time(self, exposure_time: int) -> None:
        """
        Set the exposure time of the camera.

        Args:
            exposure_time (int): Exposure time in microseconds.
                A value of 0 enables automatic exposure.

        Returns:
            None
        """
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """
        Pause the camera stream without closing or releasing the camera.

        Required by multiplexed CSI setups where the stream must be stopped
        before switching channels via GPIO/I2C, and restarted afterwards.

        Returns:
            None
        """
        pass

    @abstractmethod
    def start_stream(self) -> None:
        """
        Resume the camera stream after a channel switch or hardware change.

        Returns:
            None
        """
        pass

    @abstractmethod
    def restore_preview_settings(self) -> None:
        """
        Restore the camera to its configured preview exposure after a capture sweep.

        Called once after all capture steps in an inspection cycle so that
        subsequent preview frames are correctly lit for the web stream.

        Returns:
            None
        """
        pass
