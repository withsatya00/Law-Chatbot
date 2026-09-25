# Scheduled task entry point. The task invokes PowerShell with -WindowStyle Hidden.
# Read-only: `reconcile_state_coverage.py` never approves, verifies, re-indexes or
# mutates anything -- see docs/LAW_UPDATE_MONITORING.md.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$logDirectory = Join-Path $repoRoot 'storage/operations/state_coverage_reconciliation'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
$log = Join-Path $logDirectory 'task.log'
& $python -m scripts.reconcile_state_coverage --json 2>&1 | Out-File -LiteralPath $log -Append -Encoding utf8
exit $LASTEXITCODE
