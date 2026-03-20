@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "PROJECT_ROOT=%%~fI"

echo Launching Claude Code...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell.exe -WorkingDirectory '%PROJECT_ROOT%' -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File','%SCRIPT_DIR%\claude_launch.ps1'"
echo.
echo Claude Code launched in new window.
endlocal
