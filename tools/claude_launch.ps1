# claude_launch.ps1
# Purpose: Launch Claude Code from this workspace with optional Conda activation
# Features: Project-root cwd setup, optional environment activation, Python path echo
# Usage: powershell -ExecutionPolicy Bypass -File .\tools\claude_launch.ps1

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

$envName = Get-ProjectCondaEnv
Activate-ProjectCondaEnv -EnvName $envName

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    Write-Host "Python: $($pythonCmd.Source)"
} else {
    Write-Host "Python: not found on PATH"
}

if ($env:CONDA_DEFAULT_ENV) {
    Write-Host "Activated Conda environment: $env:CONDA_DEFAULT_ENV"
}

Set-Location $projectRoot
Write-Host "Working directory: $projectRoot"
$env:CLAUDE_CODE_AUTO_CONNECT_IDE = "false"
$env:CLAUDE_CODE_SHELL = "powershell"

# Prevent Claude from inheriting VS Code terminal markers that can trigger IDE attachment.
$vscodeEnvNames = @(
    "TERM_PROGRAM",
    "TERM_PROGRAM_VERSION",
    "VSCODE_GIT_IPC_HANDLE",
    "VSCODE_GIT_ASKPASS_EXTRA_ARGS",
    "VSCODE_GIT_ASKPASS_MAIN",
    "VSCODE_GIT_ASKPASS_NODE",
    "VSCODE_GIT_IPC_HANDLE_MAIN",
    "VSCODE_HANDLES_UNCAUGHT_ERRORS",
    "VSCODE_CODE_CACHE_PATH",
    "VSCODE_CWD",
    "VSCODE_ESM_ENTRYPOINT",
    "VSCODE_IPC_HOOK",
    "VSCODE_IPC_HOOK_CLI",
    "VSCODE_NLS_CONFIG"
)

foreach ($name in $vscodeEnvNames) {
    if (Test-Path "Env:$name") {
        Remove-Item "Env:$name"
    }
}

$claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
if (-not $claudeCmd) {
    throw "The 'claude' command is not available on PATH."
}

claude
