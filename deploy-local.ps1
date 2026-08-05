#Requires -Version 7.0
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipBackup,
    [ValidateRange(1, 365)]
    [int]$BackupKeep = 14,
    [string]$Cargo = "cargo",
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $IsWindows) {
    throw "deploy-local.ps1 supports the canonical Windows + WSL deployment path only"
}

$repoRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$athanorRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot "..\the-athanor"))
$stageTarget = Join-Path $repoRoot "target\deploy"
$stagedExe = Join-Path $stageTarget "release\solarisael-house-substrate.exe"
$stagedPdb = [IO.Path]::ChangeExtension($stagedExe, ".pdb")
$configuredLiveExe = [string]$env:SOLARISAEL_HOUSE_RUST
$liveExe = if ([string]::IsNullOrWhiteSpace($configuredLiveExe)) {
    Join-Path $repoRoot "target\release\solarisael-house-substrate.exe"
} elseif ([IO.Path]::IsPathRooted($configuredLiveExe)) {
    $configuredLiveExe
} else {
    Join-Path $repoRoot $configuredLiveExe
}
$liveExe = [IO.Path]::GetFullPath($liveExe)
$livePdb = [IO.Path]::ChangeExtension($liveExe, ".pdb")
$previousExe = Join-Path $stageTarget "previous\solarisael-house-substrate.exe"
$previousPdb = [IO.Path]::ChangeExtension($previousExe, ".pdb")

function Invoke-Checked {
    param(
        [Parameter(Mandatory)] [string]$Label,
        [Parameter(Mandatory)] [string]$FilePath,
        [Parameter(Mandatory)] [string[]]$ArgumentList
    )

    Write-Host "==> $Label"
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Get-LiveSubstrateWorkers {
    param([Parameter(Mandatory)] [string]$ExecutablePath)

    $target = [IO.Path]::GetFullPath($ExecutablePath)
    Get-CimInstance Win32_Process | Where-Object {
        if ([string]::IsNullOrWhiteSpace([string]$_.ExecutablePath)) {
            return $false
        }
        [StringComparer]::OrdinalIgnoreCase.Equals(
            [IO.Path]::GetFullPath([string]$_.ExecutablePath),
            $target
        )
    }
}

if (-not (Test-Path (Join-Path $repoRoot "Cargo.toml") -PathType Leaf)) {
    throw "substrate Cargo.toml is missing at $repoRoot"
}
if (-not (Test-Path (Join-Path $athanorRoot "Cargo.toml") -PathType Leaf)) {
    throw "The Athanor Cargo.toml is missing at $athanorRoot"
}

New-Item -ItemType Directory -Force -Path $stageTarget | Out-Null

if (-not $SkipTests) {
    Invoke-Checked -Label "Athanor core and protocol tests" -FilePath $Cargo -ArgumentList @(
        "test", "--manifest-path", (Join-Path $athanorRoot "Cargo.toml"),
        "-p", "house-core", "-p", "house-protocol"
    )
    Invoke-Checked -Label "substrate regression tests" -FilePath $Cargo -ArgumentList @(
        "test", "--manifest-path", (Join-Path $repoRoot "Cargo.toml"),
        "--release", "--target-dir", $stageTarget
    )
}

Invoke-Checked -Label "staged release build" -FilePath $Cargo -ArgumentList @(
    "build", "--manifest-path", (Join-Path $repoRoot "Cargo.toml"),
    "--release", "--target-dir", $stageTarget
)
if (-not (Test-Path $stagedExe -PathType Leaf)) {
    throw "staged executable was not produced at $stagedExe"
}

if (-not $SkipBackup) {
    $priorPgWsl = $env:SOLARISAEL_PG_WSL
    try {
        $env:SOLARISAEL_PG_WSL = "1"
        Invoke-Checked -Label "pre-deploy PostgreSQL backup" -FilePath $stagedExe -ArgumentList @(
            "backup", "--output-dir", (Join-Path $repoRoot "backups"),
            "--keep", [string]$BackupKeep
        )
    } finally {
        if ($null -eq $priorPgWsl) {
            Remove-Item Env:SOLARISAEL_PG_WSL -ErrorAction SilentlyContinue
        } else {
            $env:SOLARISAEL_PG_WSL = $priorPgWsl
        }
    }
}

$workers = @(Get-LiveSubstrateWorkers -ExecutablePath $liveExe)
if ($workers.Count -gt 0) {
    $workerSummary = ($workers | ForEach-Object { "PID=$($_.ProcessId) parent=$($_.ParentProcessId)" }) -join ", "
    Write-Host "==> stopping exact-path substrate workers: $workerSummary"
    foreach ($worker in $workers) {
        Stop-Process -Id $worker.ProcessId -Force -ErrorAction Stop
    }

    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    do {
        $remainingWorkers = @(Get-LiveSubstrateWorkers -ExecutablePath $liveExe)
        if ($remainingWorkers.Count -eq 0) {
            break
        }
        Start-Sleep -Milliseconds 200
    } while ([DateTime]::UtcNow -lt $deadline)

    if ($remainingWorkers.Count -gt 0) {
        throw "substrate workers did not stop within 10 seconds"
    }
}

New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($liveExe)) | Out-Null
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($previousExe)) | Out-Null
Remove-Item $previousExe, $previousPdb -Force -ErrorAction SilentlyContinue
if (Test-Path $liveExe -PathType Leaf) {
    Move-Item $liveExe $previousExe -Force
}
if (Test-Path $livePdb -PathType Leaf) {
    Move-Item $livePdb $previousPdb -Force
}

try {
    Copy-Item $stagedExe $liveExe -Force
    if (Test-Path $stagedPdb -PathType Leaf) {
        Copy-Item $stagedPdb $livePdb -Force
    }
    Invoke-Checked -Label "ordered database migrations" -FilePath $Python -ArgumentList @(
        (Join-Path $repoRoot "run_migrations.py")
    )
    Invoke-Checked -Label "semantic vocabulary refresh" -FilePath $liveExe -ArgumentList @(
        "semantic-vocabulary-refresh"
    )
} catch {
    Remove-Item $liveExe, $livePdb -Force -ErrorAction SilentlyContinue
    if (Test-Path $previousExe -PathType Leaf) {
        Move-Item $previousExe $liveExe -Force
    }
    if (Test-Path $previousPdb -PathType Leaf) {
        Move-Item $previousPdb $livePdb -Force
    }
    throw
}

Remove-Item $previousExe, $previousPdb -Force -ErrorAction SilentlyContinue
Invoke-Checked -Label "Full-mode health proof" -FilePath $Python -ArgumentList @(
    (Join-Path $repoRoot "health.py")
)

Write-Host "==> deployment complete"
Write-Host "live executable: $liveExe"
Write-Host "restart OMP once before the next Athanor tool call so its transport and TypeScript tool schemas reload"
