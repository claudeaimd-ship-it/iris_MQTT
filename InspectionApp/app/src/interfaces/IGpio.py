from abc import ABC, abstractmethod


class IGpio(ABC):
    """
    Abstract interface for GPIO input/output operations.

    Covers the four operations used in inspection sequences:
    - turn_on / turn_off: set an output pin high or low.
    - send_output: send a timed pulse on an output pin.
    - wait_for_input: block until an input pin reaches an expected value or a timeout elapses.

    Pin numbers and their roles (spotlight, PLC signal, trigger input, etc.)
    are defined in the sequence JSON and passed to each method at call time.
    The adapter is responsible for mapping those pin numbers to the
    underlying hardware objects.
    """

    @abstractmethod
    def turn_on(self, pin: int) -> None:
        """
        Set an output pin HIGH.

        Args:
            pin (int): GPIO pin number as defined in the sequence JSON.

        Returns:
            None
        """
        pass

    @abstractmethod
    def turn_off(self, pin: int) -> None:
        """
        Set an output pin LOW.

        Args:
            pin (int): GPIO pin number as defined in the sequence JSON.

        Returns:
            None
        """
        pass

    @abstractmethod
    def send_output(self, pin: int, value: int, duration_ms: int) -> None:
        """
        Send a timed pulse on an output pin.

        Sets the pin to the given value, waits for the specified duration,
        then resets the pin to LOW.

        Args:
            pin (int): GPIO pin number as defined in the sequence JSON.
            value (int): Output value to set (1 = HIGH, 0 = LOW).
            duration_ms (int): Duration of the pulse in milliseconds.

        Returns:
            None
        """
        pass

    @abstractmethod
    def wait_for_input(self, pin: int, expected_value: int, timeout_ms: int) -> bool:
        """
        Block until an input pin reaches the expected value or the timeout elapses.

        Args:
            pin (int): GPIO pin number as defined in the sequence JSON.
            expected_value (int): The value to wait for (1 = HIGH, 0 = LOW).
            timeout_ms (int): Maximum time to wait in milliseconds.

        Returns:
            bool: True if the expected value was detected within the timeout,
                False if the timeout elapsed without detecting it.
        """
        pass

    @abstractmethod
    def interrupt(self) -> None:
        """
        Signal any blocking ``wait_for_input`` call to return ``False`` immediately.

        Used by ``stop_loop()`` to unblock a thread that is waiting for a PLC
        trigger so the thread can exit within the next polling interval (~10 ms)
        instead of waiting up to 15 seconds for ``join()`` to time out.

        Must be paired with ``reset_interrupt()`` before the next ``start_loop()``
        so that subsequent ``wait_for_input`` calls work normally.

        Returns:
            None
        """
        pass

    @abstractmethod
    def reset_interrupt(self) -> None:
        """
        Clear the interrupt flag set by ``interrupt()``.

        Must be called from ``start_loop()`` before starting a new loop thread
        so that ``wait_for_input`` is not immediately short-circuited on the
        first call of the new cycle.

        Returns:
            None
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Release all GPIO resources.

        Should be called when the application shuts down.

        Returns:
            None
        """
        pass
