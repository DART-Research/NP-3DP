<# 
    Bootstrap script for NP Slicing on Windows.

    Creates (or recreates) the local .venv, upgrades pip, and installs packages
    from requirements.txt. The script prefers the `py` launcher when available
    and falls back to `python`/`python3`. Requires Python 3.11+.

    Usage examples:
        ./install_windows.ps1
        ./install_windows.ps1 -Recreate
        ./install_windows.ps1 -PythonExe "C:\Python311\python.exe"
#>
[CmdletBinding()]
param (
    [string]$PythonExe,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-PythonCandidate {
    param (
        [string[]]$Command
    )

    try {
        $baseArgs = @()
        if ($Command.Length -gt 1) {
            $baseArgs = $Command[1..($Command.Length - 1)]
        }
        $output = & $Command[0] @($baseArgs + @("-c", "import sys,json; print(json.dumps({'major':sys.version_info.major,'minor':sys.version_info.minor}))"))
        $info = $output | ConvertFrom-Json
        if ($info.major -eq 3 -and $info.minor -le 10) {
            return $null
        }
        if ($info.major -lt 3) {
            return $null
        }
        return [pscustomobject]@{
            Command = $Command
            Major   = $info.major
            Minor   = $info.minor
        }
    } catch {
        return $null
    }
}

function Resolve-PythonCommand {
    param (
        [string]$Override
    )

    if ($Override) {
        $candidate = @($Override)
        $result = Test-PythonCandidate -Command $candidate
        if (-not $result) {
            throw "Provided Python executable '$Override' is not a usable Python 3.11+ interpreter."
        }
        return $result.Command
    }

    $candidates = @(
        @("py", "-3.12"),
        @("py", "-3.11"),
        @("py"),
        @("python"),
        @("python3")
    )

    foreach ($candidate in $candidates) {
        $result = Test-PythonCandidate -Command $candidate
        if ($result) {
            return $result.Command
        }
    }

    throw "Unable to locate a Python 3.11+ interpreter. Install Python or provide -PythonExe."
}

$pythonCommand = Resolve-PythonCommand -Override $PythonExe

Push-Location $repoRoot
try {
    $venvPath = Join-Path $repoRoot ".venv"
    if (Test-Path $venvPath) {
        if ($Recreate) {
            Write-Host "Removing existing virtual environment at $venvPath"
            Remove-Item -Recurse -Force $venvPath
        } else {
            Write-Host ".venv already exists; it will be reused. Pass -Recreate to rebuild."
        }
    }

    if (-not (Test-Path $venvPath)) {
        Write-Host "Creating virtual environment (.venv)..."
        $pyArgs = @()
        if ($pythonCommand.Length -gt 1) {
            $pyArgs = $pythonCommand[1..($pythonCommand.Length - 1)]
        }
        & $pythonCommand[0] @($pyArgs + @("-m", "venv", ".venv"))
    }

    $venvPython = Join-Path (Join-Path $venvPath "Scripts") "python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment appears invalid (missing Scripts\python.exe)."
    }

    Write-Host "Upgrading pip..."
    & $venvPython "-m" "pip" "install" "--upgrade" "pip"

    $requirements = Join-Path $repoRoot "requirements.txt"
    if (-not (Test-Path $requirements)) {
        throw "requirements.txt not found at $requirements."
    }

    Write-Host "Installing dependencies from requirements.txt..."
    & $venvPython "-m" "pip" "install" "-r" $requirements

    Write-Host ""
    Write-Host "Virtual environment ready."
    Write-Host 'Activate it via: .\.venv\Scripts\Activate.ps1'
} finally {
    Pop-Location
}
