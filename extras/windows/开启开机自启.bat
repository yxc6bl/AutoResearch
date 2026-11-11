@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title 开启 AutoResearch 开机自启
set "PYTHONIOENCODING=gbk"

echo ==========================================
echo    开启 AutoResearch 开机自启
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

"!PY!" _tools\autostart.py on
echo.
echo 说明：只在 Windows 的「启动」文件夹里放了一个小脚本，不写注册表、不需要管理员权限。
echo       平时双击 index_session.html 就能直接用，前提就是服务在后台跑着。
echo.
pause
