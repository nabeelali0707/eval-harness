"""Tests for the duplicate-runner guard logic used by start_ablation.ps1."""
from __future__ import annotations

from pathlib import Path

SCRIPT = Path("start_ablation.ps1").read_text(encoding="utf-8")


def test_script_checks_for_existing_runner_before_starting() -> None:
    assert "run_ablation.py*" in SCRIPT
    assert "Get-CimInstance Win32_Process" in SCRIPT
    # The guard must run before the Start-Process launch.
    guard_pos = SCRIPT.index("Get-CimInstance")
    launch_pos = SCRIPT.index("Start-Process -FilePath \"python\"")
    assert guard_pos < launch_pos


def test_script_exits_without_relaunching_when_runner_exists() -> None:
    assert "exit 0" in SCRIPT[SCRIPT.index("if ($existing)"):]


def test_script_starts_ollama_when_down_and_waits_for_readiness() -> None:
    assert "Programs\\Ollama\\ollama.exe" in SCRIPT
    assert '-ArgumentList "serve"' in SCRIPT
    assert "Test-Ollama" in SCRIPT
    # Must poll for readiness instead of assuming an instant start.
    assert "for ($i = 0; $i -lt 30; $i++)" in SCRIPT


def test_script_launches_runner_detached_with_logs() -> None:
    assert "-WindowStyle Hidden" in SCRIPT
    assert "run_ablation.py --output $OutputDir" in SCRIPT
    assert "ablation_live.log" in SCRIPT
    assert "ablation_live.err" in SCRIPT


def test_script_default_output_matches_status_docs() -> None:
    assert '[string]$OutputDir = "results/final_local_ollama_150q"' in SCRIPT
