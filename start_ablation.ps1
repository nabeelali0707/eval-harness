# Relaunch the detached five-mode ablation safely.
#
# Encodes the failure modes hit in practice:
#   1. Two runners racing on the same checkpoint (duplicate-guard exits instead)
#   2. Ollama down (starts it detached and waits for readiness)
#   3. Runner dying with its parent shell (Start-Process detaches it)
#
# Usage:  powershell -ExecutionPolicy Bypass -File start_ablation.ps1
param(
    [string]$OutputDir = "results/final_local_ollama_150q"
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# 1. Never start a second runner: two runners would double the LLM load and
#    race on the same atomic checkpoint file.
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*run_ablation.py*" }
if ($existing) {
    $pids = ($existing | ForEach-Object { $_.ProcessId }) -join ", "
    Write-Host "Ablation already running (PID $pids). Nothing to do."
    Write-Host "Monitor with: python progress_report.py"
    exit 0
}

# 2. Ensure the local Ollama service is up; start it detached if not.
function Test-Ollama {
    try {
        # -UseBasicParsing avoids the IE first-run/proxy autodetect hang.
        Invoke-WebRequest -Uri "http://localhost:11434/api/tags" `
            -TimeoutSec 3 -UseBasicParsing -DisableKeepAlive | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-Ollama)) {
    $ollamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (-not (Test-Path $ollamaExe)) {
        Write-Error "Ollama is down and $ollamaExe was not found. Install Ollama first."
        exit 1
    }
    Write-Host "Ollama is down; starting it detached..."
    Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
    $up = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        if (Test-Ollama) { $up = $true; break }
    }
    if (-not $up) {
        Write-Error "Ollama did not become ready within 60 seconds."
        exit 1
    }
    Write-Host "Ollama is up."
}

# 3. Launch the runner as a detached process with logs beside the repo root.
Start-Process -FilePath "python" `
    -ArgumentList "run_ablation.py --output $OutputDir" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $root "ablation_live.log") `
    -RedirectStandardError (Join-Path $root "ablation_live.err")

Write-Host "Ablation launched detached from session $PID."
Write-Host "Monitor with: python progress_report.py"
Write-Host "Logs: ablation_live.log / ablation_live.err in $root"
