import threading
import time

from gpiozero import OutputDevice, Button

from app.src.interfaces.IGpio import IGpio


class RpiGpioAdapter(IGpio):
    """
    Adapter for GPIO input/output operations on Raspberry Pi.

    Uses gpiozero to manage output and input pins. Pins are configured
    from the sequence JSON (gpio_configuration section), which defines each
    pin number and its type ('input' or 'output').

    Pin numbers used in sequence steps are mapped at call time, so the
    adapter does not need to know the semantic role of each pin (spotlight,
    PLC signal, trigger, etc.).

    Attributes:
        _outputs (dict[int, OutputDevice]): Output pins keyed by GPIO pin number.
        _inputs (dict[int, Button]): Input pins keyed by GPIO pin number.
    """

    def __init__(self, gpio_configuration: list[dict]):
        """
        Args:
            gpio_configuration (list[dict]): Pin configuration loaded from the
                'gpio_configuration' section of the sequence JSON. Each dict must have:
                - 'pin_number' (int): GPIO BCM pin number.
                - 'type' (str): 'input' or 'output'.
        """
        self._outputs: dict[int, OutputDevice] = {}
        self._inputs: dict[int, Button] = {}
        self._closed: threading.Event = threading.Event()
        self._interrupted: threading.Event = threading.Event()

        for pin_config in gpio_configuration:
            pin = pin_config["pin_number"]
            if pin_config["type"] == "output":
                self._outputs[pin] = OutputDevice(pin, initial_value=False)
            elif pin_config["type"] == "input":
                self._inputs[pin] = Button(pin, pull_up=True)

    def turn_on(self, pin: int) -> None:
        """
        Set an output pin HIGH.

        Args:
            pin (int): GPIO pin number.

        Raises:
            ValueError: If the pin is not configured as an output.

        Returns:
            None
        """
        self._get_output(pin).on()

    def turn_off(self, pin: int) -> None:
        """
        Set an output pin LOW.

        Args:
            pin (int): GPIO pin number.

        Raises:
            ValueError: If the pin is not configured as an output.

        Returns:
            None
        """
        self._get_output(pin).off()

    def send_output(self, pin: int, value: int, duration_ms: int) -> None:
        """
        Send a timed pulse on an output pin.

        Sets the pin to the given value, waits for the specified duration,
        then resets the pin to LOW.

        Args:
            pin (int): GPIO pin number.
            value (int): Output value to set during the pulse (1 = HIGH, 0 = LOW).
            duration_ms (int): Duration of the pulse in milliseconds.

        Raises:
            ValueError: If the pin is not configured as an output.

        Returns:
            None
        """
        output = self._get_output(pin)
        output.on() if value else output.off()
        time.sleep(duration_ms / 1000)
        output.off()

    def wait_for_input(self, pin: int, expected_value: int, timeout_ms: int) -> bool:
        """
        Block until an input pin reaches the expected value or the timeout elapses.

        Polls the pin state at ~10 ms intervals.

        Args:
            pin (int): GPIO pin number.
            expected_value (int): Value to wait for (1 = pressed/HIGH, 0 = released/LOW).
            timeout_ms (int): Maximum wait time in milliseconds.

        Raises:
            ValueError: If the pin is not configured as an input.

        Returns:
            bool: True if the expected value was detected, False if timeout elapsed.
        """
        button = self._get_input(pin)
        # If timeout_ms is 0, wait indefinitely until the expected value is detected,
        # close() is called (permanent shutdown), or interrupt() is called (temporary stop).
        if not timeout_ms:
            while not self._closed.is_set() and not self._interrupted.is_set():
                current = 1 if button.is_pressed else 0
                if current == expected_value:
                    return True
                time.sleep(0.01)  # Poll at ~10 ms intervals to avoid busy-waiting.
            return False  # Closed or interrupted — signal caller to abort.

        # If timeout is more than 0, apply the timeout logic.
        deadline = time.monotonic() + (timeout_ms / 1000)
        while time.monotonic() < deadline:
            if self._closed.is_set() or self._interrupted.is_set():
                return False
            current = 1 if button.is_pressed else 0
            if current == expected_value:
                return True
            time.sleep(0.01)  # Poll at ~10 ms intervals to avoid busy-waiting.

        return False

    def interrupt(self) -> None:
        """
        Signal any blocking ``wait_for_input`` call to return ``False`` immediately.

        Sets ``_interrupted`` which is checked at each 10 ms poll interval.
        Does NOT close the GPIO hardware — call ``reset_interrupt()`` before
        the next ``start_loop()`` to allow normal operation to resume.

        Returns:
            None
        """
        self._interrupted.set()

    def reset_interrupt(self) -> None:
        """
        Clear the interrupt flag so ``wait_for_input`` operates normally again.

        Must be called from ``start_loop()`` before starting a new loop thread.

        Returns:
            None
        """
        self._interrupted.clear()

    def close(self) -> None:
        """
        Release all GPIO resources.

        Sets ``_closed`` and ``_interrupted`` before closing devices so that any
        thread blocked in ``wait_for_input`` exits cleanly rather than crashing
        with ``GPIODeviceClosed``.

        Returns:
            None
        """
        self._closed.set()
        self._interrupted.set()
        for device in self._outputs.values():
            device.close()
        for device in self._inputs.values():
            device.close()
        print("[OK] GPIO pins released.")

    # === Private helpers ===

    def _get_output(self, pin: int) -> OutputDevice:
        """Return the OutputDevice for the given pin or raise ValueError."""
        if pin not in self._outputs:
            raise ValueError(f"Pin {pin} is not configured as an output. Available outputs: {list(self._outputs.keys())}")
        return self._outputs[pin]

    def _get_input(self, pin: int) -> Button:
        """Return the Button for the given pin or raise ValueError."""
        if pin not in self._inputs:
            raise ValueError(f"Pin {pin} is not configured as an input. Available inputs: {list(self._inputs.keys())}")
        return self._inputs[pin]
