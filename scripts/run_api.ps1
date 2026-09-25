<#
.SYNOPSIS
    Starts the Legal AI Assistant FastAPI backend on this Windows machine.

.DESCRIPTION
    A thin launcher around the stock README command. All it adds is the host
    environment repairs in scripts\_win_env.ps1 (Avast's SSLKEYLOGFILE, and the
    Tesseract-vs-MSYS2 GTK DLL collision) -- read that file for why they are
    needed. No application behaviour is changed.

    Pass -Validate to run scripts\validate_environment.py first and refuse to
    start if a REQUIRED check fails (MongoDB, Redis, the configured LLM
    provider, storage, QR generation). Unavailable optional features -- OCR
    without Tesseract, say -- are reported but do not block startup, matching
    how the app itself degrades.

    One-time prerequisites, see WINDOWS_SETUP.md:
        .venv\Scripts\python.exe scripts\windows_trust_avast_ca.py
        MongoDB service running, Memurai/Redis running

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_api.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_api.ps1 -Reload
    powershell -ExecutionPolicy Bypass -File scripts\run_api.ps1 -Validate
#>
[CmdletBinding()]
param(
    [switch]$Reload,
    [switch]$Validate,
    [string]$BindHost = '0.0.0.0',
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
. "$PSScriptRoot\_win_env.ps1"

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtualenv not found at $python. Create it with: py -3.12 -m venv .venv; .venv\Scripts\python.exe -m pip install -e `".[dev]`""
}

if ($Validate) {
    Write-Host 'Validating environment...' -ForegroundColor Cyan
    & $python (Join-Path $PSScriptRoot 'validate_environment.py')
    if ($LASTEXITCODE -ne 0) {
        throw "Environment validation failed (exit $LASTEXITCODE). Fix the items marked [FAIL] above, or start without -Validate to run anyway."
    }
}

$uvicornArgs = @('-m', 'uvicorn', 'app.main:app', '--host', $BindHost, '--port', $Port)
if ($Reload) { $uvicornArgs += '--reload' }

Write-Host "Starting API on http://${BindHost}:${Port}" -ForegroundColor Cyan
& $python @uvicornArgs
exit $LASTEXITCODE
