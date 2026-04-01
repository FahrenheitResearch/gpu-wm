Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$watchDir = Join-Path $repoRoot "output\watchdog"
$localQueue = Join-Path $watchDir "local_queue.txt"
$logFile = Join-Path $watchDir "local_tick.log"
$stateFile = Join-Path $watchDir "local_tick_state.txt"
$localJobLogDir = Join-Path $watchDir "local_jobs"
$remoteHost = "root@20.119.175.17"
$remotePort = 2528

New-Item -ItemType Directory -Force -Path $watchDir | Out-Null
New-Item -ItemType Directory -Force -Path $localJobLogDir | Out-Null

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    Add-Content -Path $logFile -Value $line
    Set-Content -Path $stateFile -Value $line
}

function To-WslPath {
    param([string]$WindowsPath)
    $normalized = $WindowsPath -replace "\\", "/"
    if ($normalized.Length -ge 3 -and $normalized[1] -eq ':') {
        $drive = $normalized.Substring(0, 1).ToLower()
        $rest = $normalized.Substring(2)
        return "/mnt/$drive$rest"
    }
    return $normalized
}

function Get-LocalGpu {
    try {
        return (& nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null | Out-String).Trim()
    } catch {
        return "nvidia-smi unavailable"
    }
}

function Get-LocalRuns {
    try {
        $raw = (& wsl bash -lc "ps -eo pid,args" 2>$null | Out-String)
        $lines = $raw -split "`r?`n" | Where-Object {
            $_ -match 'build-wsl/gpu-wm|python3 tools/run_fast_case.py|python3 tools/run_gate_matrix.py|python3 tools/run_freestream_terrain.py'
        }
        return ($lines -join "`n").Trim()
    } catch {
        return ""
    }
}

function Get-RemoteSummary {
    $cmd = "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true; echo ---; ps -eo pid,args | grep -E 'build-wsl/gpu-wm|python tools/run_fast_case.py|python3 tools/run_fast_case.py' | grep -v grep || true"
    try {
        return (& ssh -p $remotePort $remoteHost $cmd 2>$null | Out-String).Trim()
    } catch {
        return "remote summary unavailable"
    }
}

function Pop-QueueCommand {
    param([string]$QueuePath)
    if (-not (Test-Path $QueuePath)) {
        return $null
    }

    $lines = Get-Content -Path $QueuePath
    $command = $null
    $remaining = New-Object System.Collections.Generic.List[string]
    foreach ($line in $lines) {
        if ($null -eq $command -and -not [string]::IsNullOrWhiteSpace($line) -and -not $line.Trim().StartsWith("#")) {
            $command = $line.Trim()
            continue
        }
        $remaining.Add($line)
    }

    if ($null -ne $command) {
        Set-Content -Path $QueuePath -Value $remaining
    }
    return $command
}

function Start-LocalExperiment {
    param([string]$Command)
    $repoWsl = To-WslPath $repoRoot
    $logName = "local_job_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
    $jobLogWin = Join-Path $localJobLogDir $logName
    $jobLogWsl = To-WslPath $jobLogWin
    $wrapped = "cd '$repoWsl' && export PYTHONUNBUFFERED=1 && ($Command) >> '$jobLogWsl' 2>&1"
    $proc = Start-Process -FilePath "wsl.exe" -ArgumentList @("bash", "-lc", $wrapped) -WindowStyle Hidden -PassThru
    Write-Log "started local experiment pid=$($proc.Id) log=$jobLogWin cmd=$Command"
}

$localRuns = Get-LocalRuns
$localGpu = Get-LocalGpu
$localUtil = 0
$localMem = 0
if ($localGpu -match ',\s*(\d+)\s*%,\s*(\d+)\s*MiB,') {
    $localUtil = [int]$Matches[1]
    $localMem = [int]$Matches[2]
}
$localBusy = (-not [string]::IsNullOrWhiteSpace($localRuns)) -or $localUtil -ge 20 -or $localMem -ge 10000
Write-Log "local busy=$localBusy gpu=[$localGpu] runs=[$localRuns]"
Write-Log "remote [$((Get-RemoteSummary))]"

if (-not $localBusy) {
    $next = Pop-QueueCommand -QueuePath $localQueue
    if ($null -ne $next -and $next.Length -gt 0) {
        Start-LocalExperiment -Command $next
    } else {
        Write-Log "local queue empty"
    }
}
