@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title 关闭 AutoResearch 开机自启
set "PYTHONIOENCODING=gbk"

echo ==========================================
echo    关闭 AutoResearch 开机自启
echo ==========================================
echo.

set "PY="
for %%D in (".venv\Scripts\python.exe" "venv\Scripts\python.exe" "env\Scripts\python.exe") do (
  if not defined PY if exist "%%~D" set "PY=%%~fD"
)
if not defined PY if exist "%USERPROFILE%\.workbuddy\binaries\python\envs\autoresearch\Scripts\python.exe" (
  set "PY=%USERPROFILE%\.workbuddy\binaries\python\envs\autoresearch\Scripts\python.exe"
)
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [错误] 没找到 Python。
  echo.
  pause
  exit /b 1
)

"!PY!" _tools\autostart.py off
echo.
pause
