import subprocess
import time

from gpiozero import OutputDevice

from app.src.interfaces.ICsiMux import ICsiMux
from app.src.interfaces.IControllableCamera import IControllableCamera


class RpiCsiMuxAdapter(ICsiMux):
    """
    Adapter for the CSI camera multiplexer used in Raspberry Pi multi-camera setups.

    Controls channel selection by combining three GPIO output pins (MUX selector)
    and an I2C command sent via the system shell. Requires the camera stream to be
    stopped before switching and restarted afterwards — this is handled internally
    by calling stop_stream() and start_stream() on the injected camera.

    The whole stop/GPIO/I2C/start sequence runs under ``camera.hw_lock`` so a
    channel switch from one thread can never interleave with a capture from
    another — Picamera2 is not safe to call concurrently across threads.

    GPIO and I2C are instantiated internally since they are fixed to the hardware
    and never change between deployments.

    Channel configuration (GPIO state and I2C command per channel) is loaded
    externally (e.g. from a JSON file) and passed at instantiation time.

    Attributes:
        _channels (dict[str, dict]): Mapping of channel name to its hardware config.
            Each entry must have 'gpio_state' (list[int], 3 values) and 'i2c_cmd' (str).
        _camera (IControllableCamera): Camera whose stream is paused during channel switch.
        _current_channel (str | None): Name of the currently active channel.
        _pin_selector_0 (OutputDevice): GPIO pin for MUX selector bit 0 (physical pin 7, GPIO 4).
        _pin_selector_1 (OutputDevice): GPIO pin for MUX selector bit 1 (physical pin 11, GPIO 17).
        _pin_selector_2 (OutputDevice): GPIO pin for MUX selector bit 2 (physical pin 12, GPIO 18).
    """
    

    def __init__(self, channels: list[dict], camera: IControllableCamera):
        """
        Args:
            channels (list[dict]): Channel configuration loaded from JSON. Each dict must contain:
                - 'name' (str): Channel identifier, e.g. 'A'.
                - 'gpio_state' (dict[str, int]): Dictionary of GPIO output states, e.g. {"4": 0, "17": 0, "18": 1}.
                - 'i2c_cmd' (str): Shell command to configure the I2C MUX for this channel.
        """
        self._channels: dict[str, dict] = {ch["name"]: ch for ch in channels}
        self._camera: IControllableCamera = camera
        self._current_channel: str | None = None

        # GPIO output pins for MUX selector — fixed to hardware, always the same.
        # The double init→close→re-init cycle forces release of any lingering
        # gpiozero handles from a previous run (e.g. process killed without shutdown).
        self._pin_selector_0 = OutputDevice(4, initial_value=False)
        self._pin_selector_1 = OutputDevice(17, initial_value=False)
        self._pin_selector_2 = OutputDevice(18, initial_value=False)
        self._pin_selector_0.close()
        self._pin_selector_1.close()
        self._pin_selector_2.close()
        self._pin_selector_0 = OutputDevice(4, initial_value=False)   # Physical pin 7,  GPIO 4
        self._pin_selector_1 = OutputDevice(17, initial_value=False)  # Physical pin 11, GPIO 17
        self._pin_selector_2 = OutputDevice(18, initial_value=False)  # Physical pin 12, GPIO 18

    def select_channel(self, channel: str, settle_delay: float = 0.0) -> None:
        """
        Switch the MUX to the specified camera channel.

        Stops the camera stream, applies the GPIO selector state and I2C command
        for the target channel, then restarts the stream.

        If the requested channel is already active, the switch is skipped.

        Args:
            channel (str): Name of the channel to select (e.g. 'A', 'B', 'C', 'D').
            settle_delay (float): Optional extra seconds to wait after restarting
                the stream before returning. ``0.0`` (default) waits none.

        Raises:
            ValueError: If the channel name is not in the configuration.
            RuntimeError: If the I2C command fails.

        Returns:
            None
        """
        if channel not in self._channels:
            raise ValueError(f"Unknown CSI MUX channel: '{channel}'. Available: {list(self._channels.keys())}")

        if channel == self._current_channel:
            return  # Already on this channel, skip unnecessary stop/start cycle.

        # Shares the camera's hw_lock so a switch can never interleave with a capture.
        with self._camera.hw_lock:
            config = self._channels[channel]
            self._camera.stop_stream()
            self._apply_gpio(config["gpio_state"])
            self._apply_i2c(config["i2c_cmd"])
            self._camera.start_stream()
            if settle_delay > 0:
                time.sleep(settle_delay)

            self._current_channel = channel

    def get_current_channel(self) -> str | None:
        """
        Return the name of the currently active channel.

        Returns:
            str | None: The active channel name, or None if no channel has been selected yet.
        """
        return self._current_channel

    def select_channel_gpio_only(self, channel: str) -> None:
        """
        Switch the MUX GPIO/I2C state without touching the camera stream.

        Used only by ``scripts/cold_boot_camera_prewarm.py`` (a disposable
        per-channel subprocess run before Iris starts), where the caller
        performs a full ``initialize_camera()`` right after the switch instead
        of the normal stop_stream/start_stream cycle — there is no running
        stream to pause on a fresh per-channel reinit.

        Unlike ``select_channel()``, this never skips the hardware switch even
        if ``channel`` matches ``_current_channel`` — the cold-boot sweep must
        force a real switch on every channel regardless of prior state.

        Args:
            channel (str): Name of the channel to select (e.g. 'A', 'B', 'C', 'D').

        Raises:
            ValueError: If the channel name is not in the configuration.
            RuntimeError: If the I2C command fails.

        Returns:
            None
        """
        if channel not in self._channels:
            raise ValueError(f"Unknown CSI MUX channel: '{channel}'. Available: {list(self._channels.keys())}")

        with self._camera.hw_lock:
            config = self._channels[channel]
            self._apply_gpio(config["gpio_state"])
            self._apply_i2c(config["i2c_cmd"])
            self._current_channel = channel

    def close(self) -> None:
        """
        Release all GPIO resources.

        Should be called when the application shuts down to free the GPIO pins.

        Returns:
            None
        """
        self._pin_selector_0.close()
        self._pin_selector_1.close()
        self._pin_selector_2.close()

    def reinitialize(self) -> None:
        """
        Release and re-acquire the GPIO selector pins, and forget the current
        channel, without replacing this adapter instance.

        Runs the same double init→close→re-init cycle used at construction
        time to clear any stale ``gpiozero`` handles. Resetting
        ``_current_channel`` to ``None`` forces the next ``select_channel()``
        call to re-apply the GPIO state and I2C command instead of skipping
        the switch because the channel "looks" unchanged.

        Returns:
            None
        """
        self._pin_selector_0.close()
        self._pin_selector_1.close()
        self._pin_selector_2.close()
        self._pin_selector_0 = OutputDevice(4, initial_value=False)   # Physical pin 7,  GPIO 4
        self._pin_selector_1 = OutputDevice(17, initial_value=False)  # Physical pin 11, GPIO 17
        self._pin_selector_2 = OutputDevice(18, initial_value=False)  # Physical pin 12, GPIO 18
        self._current_channel = None

    # === Private helpers ===

    def _apply_gpio(self, gpio_state: list[int]) -> None:
        """Apply the 3-bit GPIO selector state for the target channel."""
        # Force a neutral state first so every switch is a real level transition, never a no-op re-write.
        self._pin_selector_0.off()
        self._pin_selector_1.off()
        self._pin_selector_2.off()
        time.sleep(0.02)
        self._pin_selector_0.on() if gpio_state.get("4", 0) else self._pin_selector_0.off()
        self._pin_selector_1.on() if gpio_state.get("17", 0) else self._pin_selector_1.off()
        self._pin_selector_2.on() if gpio_state.get("18", 0) else self._pin_selector_2.off()

    def _apply_i2c(self, i2c_cmd: str) -> None:
        """Execute the I2C shell command to configure the MUX chip for the target channel."""
        # Readback verification was tried and reverted: register 0x00 on this
        # chip does not reliably reflect the value just written (false positives
        # even on channel A), so exit code is the only trustworthy signal here.
        result = subprocess.run(i2c_cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"I2C command failed (exit code {result.returncode}): {i2c_cmd} — {result.stderr.strip()}")
