"""Tests for the sleep-prevention utility."""
from __future__ import annotations

import subprocess

from src.keep_awake import (
    CaffeinateInhibitor,
    NullInhibitor,
    WindowsInhibitor,
    inhibit_sleep,
)


class RecordingStateCall:
    def __init__(self, return_value: int = 0x80000001) -> None:
        self.calls: list[int] = []
        self.return_value = return_value

    def __call__(self, flags: int) -> int:
        self.calls.append(flags)
        return self.return_value


def test_windows_inhibitor_sets_and_resets_state() -> None:
    recorder = RecordingStateCall()
    inhibitor = WindowsInhibitor(state_call=recorder)

    assert inhibitor.start() is True
    assert recorder.calls[-1] == 0x80000000 | 0x00000001 | 0x00000002

    inhibitor.stop()
    assert recorder.calls[-1] == 0x80000000


def test_windows_inhibitor_reports_failure() -> None:
    inhibitor = WindowsInhibitor(state_call=RecordingStateCall(return_value=0))
    assert inhibitor.start() is False


def test_caffeinate_inhibitor_spawns_and_terminates_process(monkeypatch) -> None:
    monkeypatch.setattr("src.keep_awake.shutil.which", lambda name: "/usr/bin/caffeinate")
    started: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            started.append(command)
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int | None = None) -> int:
            return 0

    inhibitor = CaffeinateInhibitor(popen=FakeProcess)
    assert inhibitor.start() is True
    assert started == [["caffeinate", "-ims"]]

    inhibitor.stop()
    assert inhibitor._process is None


def test_caffeinate_inhibitor_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr("src.keep_awake.shutil.which", lambda name: None)
    inhibitor = CaffeinateInhibitor(popen=lambda *a, **k: subprocess.Popen(["true"]))
    assert inhibitor.start() is False


def test_null_inhibitor_is_never_active() -> None:
    inhibitor = NullInhibitor()
    assert inhibitor.start() is False
    inhibitor.stop()


def test_inhibit_sleep_yields_active_flag() -> None:
    with inhibit_sleep(NullInhibitor()) as active:
        assert active is False
