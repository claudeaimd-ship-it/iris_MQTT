import datetime
import os
import sys


class _TeeStream:
    """
    File-like wrapper that writes to both the original stream and a log file,
    prepending a timestamp to every non-empty line.

    An internal line buffer handles the case where a single ``write()`` call
    delivers a partial line (common with ``print()`` internals).

    Args:
        original: The original stream being replaced (e.g. ``sys.stdout``).
        get_log_file (callable): Zero-argument callable that returns the current
            log file handle (or ``None`` if the logger is inactive).  The
            callable is invoked on every write so that day-rotation is handled
            transparently.
    """

    def __init__(self, original, get_log_file):
        self._original     = original
        self._get_log_file = get_log_file
        self._buffer       = ""

    def write(self, msg: str) -> None:
        lines = (self._buffer + msg).split("\n")

        for line in lines[:-1]:
            if line.strip():
                now = datetime.datetime.now()
                ts  = now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 1000:03d}"
                out = f"{ts} - {line}\n"
            else:
                out = "\n"

            self._original.write(out)
            log = self._get_log_file()
            if log is not None:
                log.write(out)
                log.flush()

        self._buffer = lines[-1]

    def flush(self) -> None:
        self._original.flush()
        log = self._get_log_file()
        if log is not None:
            log.flush()

    def fileno(self) -> int:
        return self._original.fileno()


class TimestampedFileLogger:
    """
    Redirects ``sys.stdout`` and ``sys.stderr`` to simultaneously write to the
    terminal and a daily-rotating log file.

    Log files are organised as::

        {logs_path}/YYYYMM/YYYYMMDD_log-{suffix}.txt

    Rotation happens automatically on write when the calendar day changes, so
    a process that runs past midnight seamlessly creates the next day's file
    without requiring a restart.

    Both ``stdout`` and ``stderr`` are captured, so all ``print()`` calls and
    unhandled exception tracebacks appear in the log with timestamps.

    Usage (context manager — recommended)::

        with TimestampedFileLogger("./logs", "inspection"):
            # all prints go to both console and log file
            ...

    Usage (manual — for long-running services)::

        logger = TimestampedFileLogger("./logs", "inspection")
        logger.start()
        try:
            run_application()
        finally:
            logger.stop()

    Attributes:
        _logs_path (str): Root directory for log files.
        _suffix (str): Label appended to each log filename.
        _log_file: Open file handle for the current log file, or ``None``.
        _current_log_path (str): Path of the file currently being written.
        _original_stdout: Saved reference to the real ``sys.stdout``.
        _original_stderr: Saved reference to the real ``sys.stderr``.
        _active (bool): True while logging is active.
    """

    def __init__(self, logs_path: str, suffix: str = "inspection"):
        """
        Args:
            logs_path (str): Root directory for log files.  Created
                automatically when ``start()`` is called.
            suffix (str): Label appended to each log filename.  For example,
                ``"inspection"`` produces ``YYYYMMDD_log-inspection.txt``.
        """
        self._logs_path         = logs_path
        self._suffix            = suffix
        self._log_file          = None
        self._current_log_path  = ""
        self._original_stdout   = None
        self._original_stderr   = None
        self._active            = False

    # =========================================================================
    # Public API
    # =========================================================================

    def start(self) -> None:
        """
        Redirect ``sys.stdout`` and ``sys.stderr`` to the tee streams.

        Creates the log directory and opens the log file.  Safe to call
        multiple times — subsequent calls while already active are no-ops.

        Returns:
            None
        """
        if self._active:
            return

        self._open_log_file()
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _TeeStream(self._original_stdout, self._get_log_file)
        sys.stderr = _TeeStream(self._original_stderr, self._get_log_file)
        self._active = True
        print(f"[INFO] TimestampedFileLogger: logging to '{self._current_log_path}'")

    def stop(self) -> None:
        """
        Restore ``sys.stdout`` / ``sys.stderr`` to their original values and
        close the log file.

        Safe to call even if the logger was never started.

        Returns:
            None
        """
        if not self._active:
            return

        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self._active = False

    @property
    def current_log_path(self) -> str:
        """Absolute path of the file currently being written to."""
        return self._current_log_path

    # =========================================================================
    # Context manager
    # =========================================================================

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # =========================================================================
    # Private — file management
    # =========================================================================

    def _get_expected_log_path(self) -> str:
        now       = datetime.datetime.now()
        month_dir = os.path.join(self._logs_path, now.strftime("%Y%m"))
        return os.path.join(month_dir, now.strftime(f"%Y%m%d_log-{self._suffix}.txt"))

    def _open_log_file(self) -> None:
        path = self._get_expected_log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._log_file         = open(path, "a", encoding="utf-8")
        self._current_log_path = path

    def _get_log_file(self):
        """
        Return the current log file handle, rotating to a new file if the
        calendar day (or month) has changed since the last write.

        Returns:
            file | None: Open log file handle, or ``None`` if inactive.
        """
        if not self._active:
            return None
        expected = self._get_expected_log_path()
        if expected != self._current_log_path:
            if self._log_file is not None:
                self._log_file.close()
            self._open_log_file()
        return self._log_file
