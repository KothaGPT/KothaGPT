param(
    [switch]$DryRun,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

$steps = @()

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    $steps += "winget install -e --id Python.Python.3.12"
}
if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    $steps += "winget install -e --id GoLang.Go"
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $steps += "winget install -e --id Rustlang.Rustup"
}

$steps += @(
    "py -m venv .venv",
    ". .venv\Scripts\Activate.ps1; python -m pip install --upgrade pip",
    ". .venv\Scripts\Activate.ps1; python -m pip install -r services/api/requirements.txt",
    "corepack enable",
    "pnpm install"
)

if (Test-Path ".venv\Scripts\python.exe") {
    $steps = $steps[1..($steps.Count - 1)]
}

function Show-Preview {
    Write-Host "==> Kotha GPT bootstrap preview - windows"
    $i = 1
    foreach ($s in $steps) {
        Write-Host ("  {0,2}. {1}" -f $i, $s)
        $i++
    }
}

Show-Preview

if ($DryRun) {
    exit 0
}

if (-not $Yes) {
    if (-not [Environment]::UserInteractive) {
        Write-Error "non-interactive session; re-run with -Yes to execute"
        exit 1
    }
    $reply = Read-Host "Proceed with bootstrap? [y/N]"
    if ($reply -notmatch '^[Yy]$') {
        Write-Error "aborted"
        exit 1
    }
}

foreach ($s in $steps) {
    Write-Host "==> + $s"
    Invoke-Expression $s
}

Write-Host "==> Kotha GPT bootstrap complete."
