@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title 停止 AutoResearch 网页服务

echo ==========================================
echo    停止 AutoResearch 网页服务
echo ==========================================
echo.

set "FOUND="
for %%P in (5055 5056 5060 8080) do (
  for /f "tokens=5" %%A in ('netstat -ano ^| findstr /c:":%%P " ^| findstr /i "LISTENING"') do (
    for /f "tokens=1 delims=," %%N in ('tasklist /FI "PID eq %%A" /FO CSV /NH 2^>nul') do (
      echo %%N | findstr /i "python" >nul && set "FOUND=!FOUND! %%P:%%A"
    )
  )
)

if not defined FOUND (
  echo 5055 / 5056 / 5060 / 8080 上没有 python 服务在跑，不用处理。
  echo.
  pause
  exit /b 0
)

for %%X in (!FOUND!) do (
  for /f "tokens=1,2 delims=:" %%P in ("%%X") do (
    echo 停止端口 %%P 上的服务（PID %%Q）...
    taskkill /F /PID %%Q >nul 2>nul
    if errorlevel 1 (
      echo   [失败] 杀不掉，可能权限不足，试试以管理员身份运行这个脚本。
    ) else (
      echo   [完成] 已停止。
    )
  )
)

echo.
echo 提示：开机自启还在，下次登录会自己起来。要彻底关掉，双击「关闭开机自启.bat」。
echo.
pause
