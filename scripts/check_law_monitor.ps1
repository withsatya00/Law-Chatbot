# Scheduled task entry point. The task invokes PowerShell with -WindowStyle Hidden.
$ErrorActionPreference = 'Stop'
$monitorRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $monitorRoot
$monitorLogDirectory = Join-Path $monitorRoot 'storage/law_monitor'
New-Item -ItemType Directory -Path $monitorLogDirectory -Force | Out-Null
$monitorPython = Join-Path $monitorRoot '.venv/Scripts/python.exe'
$monitorLog = Join-Path $monitorLogDirectory 'worker.log'
& $monitorPython -m scripts.run_law_monitor --once 2>&1 | Out-File -LiteralPath $monitorLog -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $monitorPython -m scripts.run_kb_machine_verification 2>&1 | Out-File -LiteralPath $monitorLog -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $monitorPython -m scripts.sync_official_kb_sources --if-due-hours 24 2>&1 | Out-File -LiteralPath $monitorLog -Append -Encoding utf8
exit $LASTEXITCODE
