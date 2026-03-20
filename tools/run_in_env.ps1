# run_in_env.ps1
# Purpose: Run a project Python entry point from the workspace root
# Features: Optional Conda activation, root cwd setup, script-path validation
# Usage: powershell -ExecutionPolicy Bypass -File .\tools\run_in_env.ps1 src\main.py [args...]

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

function Get-ProjectCondaEnv {
    $envFile = Join-Path $projectRoot ".conda-env.txt"
    if (Test-Path $envFile) {
        $envName = (Get-Content $envFile | Select-Object -First 1).Trim()
        if ($envName) {
            return $envName
        }
    }
    if ($env:PROJECT_CONDA_ENV) {
        return $env:PROJECT_CONDA_ENV
    }
    return "drum310"
}

function Activate-ProjectCondaEnv {
    param([string]$EnvName)

    if (-not $EnvName) {
        Write-Host "No Conda environment configured. Using current shell environment."
        return
    }

    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $condaCmd) {
        throw "Conda was requested via '$EnvName' but the 'conda' command is not available."
    }

    & conda 'shell.powershell' 'hook' | Out-String | Invoke-Expression
    conda activate $EnvName

    if ($env:CONDA_DEFAULT_ENV -ne $EnvName) {
        throw "Failed to activate Conda environment '$EnvName'."
    }
}

if ($args.Count -lt 1) {
    throw "Usage: run_in_env.ps1 <script_path> [args...]"
}

$scriptPath = $args[0]
$scriptArgs = @()
if ($args.Count -gt 1) {
    $scriptArgs = $args[1..($args.Count - 1)]
}

$resolvedScriptPath = Join-Path $projectRoot $scriptPath
if (-not (Test-Path $resolvedScriptPath)) {
    throw "Missing script: $resolvedScriptPath"
}

$envName = Get-ProjectCondaEnv
Activate-ProjectCondaEnv -EnvName $envName

Set-Location $projectRoot

$pythonCmd = Get-Command python -ErrorAction Stop
Write-Host "Python: $($pythonCmd.Source)"
if ($env:CONDA_DEFAULT_ENV) {
    Write-Host "Activated Conda environment: $env:CONDA_DEFAULT_ENV"
}
Write-Host "Working directory: $projectRoot"
Write-Host "Running: python $scriptPath $scriptArgs"

& python $resolvedScriptPath @scriptArgs
