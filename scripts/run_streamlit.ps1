<#
.SYNOPSIS
    Starts the Streamlit developer console. Start the API first.

.DESCRIPTION
    Thin launcher around the stock README command, plus the host environment
    repairs in scripts\_win_env.ps1. No application behaviour is changed.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1
#>
[CmdletBinding()]
param(
    [int]$Port = 8501,
    [string]$ApiBaseUrl = 'http://localhost:8000'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
. "$PSScriptRoot\_win_env.ps1"

$env:API_BASE_URL = $ApiBaseUrl

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtualenv not found at $python. Create it with: py -3.12 -m venv .venv; .venv\Scripts\python.exe -m pip install -e `".[dev]`""
}

Write-Host "Starting Streamlit console on http://localhost:$Port (API: $ApiBaseUrl)" -ForegroundColor Cyan
& $python -m streamlit run streamlit_app/app.py --server.port $Port
exit $LASTEXITCODE
