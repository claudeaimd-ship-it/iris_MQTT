import time

from app.src.interfaces.IGpio import IGpio


class NullGpioAdapter(IGpio):
    """
    Null-object GPIO adapter for platforms without physical GPIO hardware.

    Intended for use on PC or any device where GPIO is not available or not
    yet implemented. All output operations are no-ops. ``wait_for_input``
    simulates an immediate trigger (returns ``True``) so that sequences
    progress without hanging.

    A ``[SIM]`` prefix is printed for every operation so the operator can
    distinguish simulated I/O from real hardware activity.
    """

    def __init__(self, gpio_configuration: list[dict]):
        """
        Args:
            gpio_configuration (list[dict]): Accepted but ignored.
                Provided for drop-in compatibility with ``RpiGpioAdapter``.
        """
        pins = [p["pin_number"] for p in gpio_configuration]
        print(f"[SIM] NullGpioAdapter: GPIO simulation active. Pins declared: {pins}")

    # =========================================================================
    # IGpio interface
    # =========================================================================

    def turn_on(self, pin: int) -> None:
        """No-op. Logs the operation."""
        print(f"[SIM] GPIO turn_on  pin={pin}")

    def turn_off(self, pin: int) -> None:
        """No-op. Logs the operation."""
        print(f"[SIM] GPIO turn_off pin={pin}")

    def send_output(self, pin: int, value: int, duration_ms: int) -> None:
        """Simulates a timed pulse by sleeping for ``duration_ms``."""
        print(f"[SIM] GPIO send_output pin={pin} value={value} duration={duration_ms} ms")
        time.sleep(duration_ms / 1000)

    def wait_for_input(self, pin: int, expected_value: int, timeout_ms: int) -> bool:
        """
        Simulates an immediate successful trigger (always returns ``True``).

        Args:
            pin (int): GPIO pin number.
            expected_value (int): The value being waited for.
            timeout_ms (int): Ignored in simulation.

        Returns:
            bool: Always ``True``.
        """
        print(f"[SIM] GPIO wait_for_input pin={pin} expected={expected_value} → OK (simulated)")
        return True

    def interrupt(self) -> None:
        """No-op — NullGpioAdapter never blocks."""

    def reset_interrupt(self) -> None:
        """No-op — NullGpioAdapter never blocks."""

    def close(self) -> None:
        """No-op."""
        print("[SIM] NullGpioAdapter closed.")
