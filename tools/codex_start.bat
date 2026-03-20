@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "PROJECT_ROOT=%%~fI"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%i"

set "LOG_DIR=%PROJECT_ROOT%\codex_logs"
set "LOG_FILE=%LOG_DIR%\codex_session_%TIMESTAMP%.txt"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo Codex Code Session - %date% %time% > "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"
echo Project Root: %PROJECT_ROOT% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
echo Notes: >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"
echo End of session >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"
echo Created placeholder log file: %LOG_FILE%
echo.

start notepad "%LOG_FILE%"

echo Launching Codex Code...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell.exe -WorkingDirectory '%PROJECT_ROOT%' -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File','%SCRIPT_DIR%\codex_launch.ps1'"
echo.
echo Codex Code launched in new window. Notepad is open for session notes.
endlocal
