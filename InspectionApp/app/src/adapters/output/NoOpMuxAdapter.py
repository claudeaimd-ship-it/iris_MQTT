from app.src.interfaces.ICsiMux import ICsiMux


class NoOpMuxAdapter(ICsiMux):
    """
    Null-object implementation of ``ICsiMux`` for camera setups without a
    hardware MUX (e.g. single USB cameras, single IP cameras, or direct
    single-CSI connections).

    Every method is a safe no-op: ``select_channel()`` does nothing because
    the camera is always on a single implicit "channel", and ``close()``
    has nothing to release.

    This adapter allows ``SampleCaptureService`` and ``SequenceExecutor``
    to call ``mux.select_channel()`` unconditionally without branching on
    whether a physical MUX is present. The decision lives only in
    ``AppFactory._build_hardware()``, which injects this adapter whenever
    the sequence JSON declares a non-CSI-MUX camera type.

    Attributes:
        _channel (str | None): Last channel name passed to ``select_channel()``.
            Tracked so ``get_current_channel()`` returns a meaningful value.
    """

    def __init__(self) -> None:
        self._channel: str | None = None

    def select_channel(self, channel: str, settle_delay: float = 0.0) -> None:
        """
        Record the channel name and return immediately — no hardware action.

        Args:
            channel (str): Channel name (stored for ``get_current_channel()``).
            settle_delay (float): Ignored — no hardware switch happens here.

        Returns:
            None
        """
        self._channel = channel

    def get_current_channel(self) -> str | None:
        """
        Return the last channel name passed to ``select_channel()``.

        Returns:
            str | None: Last requested channel, or ``None`` if never called.
        """
        return self._channel

    def select_channel_gpio_only(self, channel: str) -> None:
        """Record the channel name and return immediately — no hardware action."""
        self._channel = channel

    def close(self) -> None:
        """No-op — nothing to release."""
        pass

    def reinitialize(self) -> None:
        """No-op — no hardware to recover; keeps the interface contract."""
        pass
