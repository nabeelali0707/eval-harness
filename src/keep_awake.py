"""Prevent OS sleep during long-running local evaluations.

Windows uses ``SetThreadExecutionState`` with ``ES_CONTINUOUS``, macOS runs
``caffeinate`` as a child process, and any other platform falls back to a
no-op inhibitor. On Windows the inhibition is scoped to the calling thread,
so the inhibitor must stay alive on a long-lived thread for the whole run.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from contextlib import contextmanager
from typing import Callable, Iterator, Protocol

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


class Inhibitor(Protocol):
    def start(self) -> bool: ...

    def stop(self) -> None: ...


class NullInhibitor:
    """No-op inhibitor for platforms without supported sleep prevention."""

    def start(self) -> bool:
        return False

    def stop(self) -> None:
        return None


class WindowsInhibitor:
    """Hold ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED."""

    def __init__(self, state_call: Callable[[int], int] | None = None) -> None:
        if state_call is None:
            def state_call(flags: int) -> int:
                return int(
                    ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]
                )

        self._state_call = state_call
        self._active = False

    def start(self) -> bool:
        previous = self._state_call(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        self._active = previous != 0
        return self._active

    def stop(self) -> None:
        self._state_call(ES_CONTINUOUS)
        self._active = False


class CaffeinateInhibitor:
    """macOS: keep ``caffeinate -ims`` running as a child process."""

    def __init__(self, popen: Callable[..., subprocess.Popen] | None = None) -> None:
        self._popen = popen or subprocess.Popen
        self._process: subprocess.Popen | None = None

    def start(self) -> bool:
        if shutil.which("caffeinate") is None:
            return False
        self._process = self._popen(["caffeinate", "-ims"])
        return self._process is not None

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None


def create_inhibitor() -> Inhibitor:
    if sys.platform == "win32":
        return WindowsInhibitor()
    if sys.platform == "darwin":
        return CaffeinateInhibitor()
    return NullInhibitor()


@contextmanager
def inhibit_sleep(inhibitor: Inhibitor | None = None) -> Iterator[bool]:
    """Prevent sleep while the context is active.

    Yields whether sleep prevention is actually active so callers can warn
    when the platform does not support it.
    """
    inhibitor = inhibitor or create_inhibitor()
    active = False
    try:
        active = inhibitor.start()
        if not active:
            print(
                "WARNING: sleep prevention is not available on this platform; "
                "keep the machine awake manually during long runs.",
                file=sys.stderr,
            )
        yield active
    finally:
        inhibitor.stop()
