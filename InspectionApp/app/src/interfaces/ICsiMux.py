from abc import ABC, abstractmethod


class ICsiMux(ABC):
    """
    Abstract interface for a CSI camera multiplexer.

    A CSI MUX allows a single CSI port to switch between multiple camera
    modules (channels). Switching requires stopping the camera stream,
    applying the hardware configuration (GPIO state + I2C command), and
    restarting the stream.

    Channel configuration (GPIO state and I2C command per channel) is
    expected to be loaded from an external source (e.g. a JSON file) and
    passed at instantiation time.
    """

    @abstractmethod
    def select_channel(self, channel: str, settle_delay: float = 0.0) -> None:
        """
        Switch the MUX to the specified camera channel.

        Implementations must stop the camera stream, apply the GPIO and I2C
        configuration for the target channel, and restart the stream.

        If the requested channel is already active, the implementation may
        skip the hardware switch to avoid unnecessary stop/start cycles.

        Args:
            channel (str): Name of the channel to select (e.g. 'A', 'B', 'C', 'D').
            settle_delay (float): Optional extra seconds to wait after restarting
                the stream before returning. ``0.0`` (default) waits none — used
                by regular preview/production channel switches. Only diagnostic
                callers (e.g. a startup warm-up pass) should pass a value > 0.

        Raises:
            ValueError: If the channel name is not recognized.
            RuntimeError: If the hardware switch fails.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_current_channel(self) -> str | None:
        """
        Return the name of the currently active channel.

        Returns:
            str | None: The active channel name, or None if no channel has
                been selected yet.
        """
        pass

    @abstractmethod
    def select_channel_gpio_only(self, channel: str) -> None:
        """
        Switch only the GPIO/I2C hardware state for the given channel, without
        touching the camera stream (no ``stop_stream()``/``start_stream()`` calls).

        Used by ``scripts/cold_boot_camera_prewarm.py`` (a disposable per-channel
        subprocess run before Iris starts), where the caller performs a full
        camera reinitialization immediately after the switch instead of the
        normal stream pause/resume cycle used by ``select_channel()``. Unlike
        ``select_channel()``, implementations must never skip the hardware
        switch based on the last selected channel.

        Args:
            channel (str): Name of the channel to select (e.g. 'A', 'B', 'C', 'D').

        Raises:
            ValueError: If the channel name is not recognized.
            RuntimeError: If the hardware switch fails.

        Returns:
            None
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Release any hardware resources held by the MUX.

        Safe to call even if no channel has been selected. Must be idempotent.

        Returns:
            None
        """
        pass

    @abstractmethod
    def reinitialize(self) -> None:
        """
        Recover the MUX hardware in place after a suspected fault, without
        discarding this adapter instance.

        Implementations must forget the currently selected channel (so the
        next ``select_channel()`` call always re-applies the GPIO/I2C
        configuration instead of skipping it as a no-op) and release/reacquire
        any hardware handles that could have gone stale.

        Returns:
            None
        """
        pass
