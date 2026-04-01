param(
    [string]$TaskName = "gpuwm-watchdog-tick",
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$tickScript = Join-Path $RepoRoot "tools\ops\local_worker_tick.ps1"
if (-not (Test-Path $tickScript)) {
    throw "tick script not found: $tickScript"
}

$taskRun = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}" -RepoRoot "{1}"' -f $tickScript, $RepoRoot
$null = & schtasks.exe /Query /TN $TaskName 2>$null
if ($LASTEXITCODE -eq 0) {
    $null = & schtasks.exe /Change /TN $TaskName /TR $taskRun
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks.exe failed while updating task '$TaskName'"
    }
} else {
    $null = & schtasks.exe /Create /F /SC MINUTE /MO 10 /TN $TaskName /TR $taskRun
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks.exe failed while creating task '$TaskName'"
    }
}

Write-Host "Installed scheduled task '$TaskName' for $tickScript"
