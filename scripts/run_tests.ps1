<#
.SYNOPSIS
    Runs the project's pytest suite with the host environment repairs applied.

.DESCRIPTION
    Equivalent to `pytest` (testpaths and asyncio_mode come from
    pyproject.toml); it only adds scripts\_win_env.ps1 first, without which
    14 test modules fail at collection on `import app.main`.

    Before running, checks whether a backend (uvicorn app.main:app) is
    already listening on the configured API port. The KB schedulers
    (kb_automation, gap_autofetch, official_source_sync, machine_verification)
    all start in-process inside that backend's FastAPI lifespan (app\main.py),
    so a live backend means those schedulers are live too, writing to the
    same storage/DB the test suite exercises -- which previously produced a
    false storage-teardown failure that took timestamp correlation to prove
    wasn't caused by pytest itself. Refuses to run against a live backend
    unless -Force is passed.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1 -PytestArgs '-k','drafting','-v'
    powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1 -Force
#>
[CmdletBinding()]
param(
    [string[]]$PytestArgs = @('-q'),
    # `%TEMP%\pytest-of-<user>` has a broken ACL on some machines here
    # (`Get-Acl` itself raises UnauthorizedAccessException), which makes pytest
    # fail during startup for a reason unrelated to any test. A project-local
    # basetemp sidesteps it without touching OS-level ACLs. Defaulted in the
    # body rather than here: `$PSScriptRoot` is not yet bound while parameter
    # defaults are evaluated, so referencing it here failed the script outright.
    # Pass '' explicitly to let pytest choose its own.
    [string]$BaseTemp,
    # Skip the running-backend/KB-scheduler preflight check.
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
. "$PSScriptRoot\_win_env.ps1"

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtualenv not found at $python. Create it with: py -3.12 -m venv .venv; .venv\Scripts\python.exe -m pip install -e `".[dev]`""
}

if (-not $Force) {
    $apiPort = 8000
    $envFile = Join-Path $projectRoot '.env'
    if (Test-Path -LiteralPath $envFile) {
        $portLine = Select-String -LiteralPath $envFile -Pattern '^API_PORT=(\d+)' | Select-Object -First 1
        if ($portLine) { $apiPort = [int]$portLine.Matches[0].Groups[1].Value }
    }

    $tcpClient = New-Object System.Net.Sockets.TcpClient
    $backendIsRunning = $false
    try {
        $connectTask = $tcpClient.ConnectAsync('127.0.0.1', $apiPort)
        $backendIsRunning = $connectTask.Wait(500) -and $tcpClient.Connected
    } catch {
        $backendIsRunning = $false
    } finally {
        $tcpClient.Close()
    }

    if ($backendIsRunning) {
        throw "A backend is already listening on 127.0.0.1:$apiPort -- its in-process KB schedulers (kb_automation, gap_autofetch, official_source_sync, machine_verification) will race pytest's storage teardown and can produce a false storage-leak failure. Stop the backend first, or pass -Force to run anyway."
    }
}

if (-not $PSBoundParameters.ContainsKey('BaseTemp')) {
    $BaseTemp = Join-Path $projectRoot '.pytest-local\basetemp'
}

$arguments = @($PytestArgs)
if ($BaseTemp) {
    New-Item -ItemType Directory -Force -Path $BaseTemp | Out-Null
    $arguments += "--basetemp=$BaseTemp"
}

& $python -m pytest @arguments
exit $LASTEXITCODE
