@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem 用法：
rem   不带参数  = 交互启动：起服务 + 自动开浏览器（日常双击这个）
rem   service   = 静默启动：不开浏览器、不 pause、输出写 logs\web.log（开机自启用这个）
set "SILENT="
if /i "%~1"=="service" set "SILENT=1"

if defined SILENT (
  title AutoResearch 服务
  if not exist "logs" mkdir "logs"
  rem 两个日志文件：service.log 记启动器动作（本脚本写，随时可写）；
  rem web.log 记服务输出（被 python 独占，所以不混在一起写）。
  set "LOG=logs\service.log"
  set "RUNLOG=logs\web.log"
  if exist "!LOG!" for %%A in ("!LOG!") do if %%~zA GTR 262144 move /y "!LOG!" "logs\service.old.log" >nul 2>nul
  if exist "!RUNLOG!" for %%A in ("!RUNLOG!") do if %%~zA GTR 1048576 move /y "!RUNLOG!" "logs\web.old.log" >nul 2>nul
  rem 日志统一 UTF-8：本脚本写的都是英文提示行，避免 GBK 与 UTF-8 混写
  set "PYTHONIOENCODING=utf-8"
  set "PYTHONUTF8=1"
) else (
  title AutoResearch 网页服务
  echo ==========================================
  echo    AutoResearch 网页服务启动器
  echo ==========================================
  echo.
)

rem —— 1. 找一个能跑实验的 Python（优先项目自带虚拟环境）——
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
  if defined SILENT (
    echo [!TIME!] ERROR python not found, server not started>>"!LOG!"
    exit /b 1
  )
  echo [错误] 没找到 Python，请先安装 Python 3.10 以上版本。
  echo.
  pause
  exit /b 1
)
if not defined SILENT echo [1/3] 使用解释器：!PY!

"!PY!" -c "import sklearn" >nul 2>nul
if errorlevel 1 (
  if defined SILENT (
    echo [!TIME!] WARNING sklearn missing, experiments will fail>>"!LOG!"
  ) else (
    echo [警告] 这个 Python 没装 sklearn，实验阶段会失败。
    echo        装一下： "!PY!" -m pip install -r requirements.txt
    echo.
  )
)

rem —— 2. 5055 上已经有服务就直接复用，不重复起 ——
"!PY!" -c "import urllib.request,sys;urllib.request.urlopen('http://127.0.0.1:5055/api/health',timeout=2);sys.exit(0)" >nul 2>nul
if !errorlevel!==0 (
  if defined SILENT (
    echo [!TIME!] already running on 5055, skip>>"!LOG!"
    exit /b 0
  )
  echo [2/3] 5055 端口已经有服务在跑，直接打开浏览器
  start "" "http://127.0.0.1:5055/"
  echo.
  echo 完成。如果页面显示不出来，刷新一次（F5）即可。
  ping -n 4 127.0.0.1 >nul
  exit /b 0
)

set "PORT="
for %%P in (5055 5056 5060 8080) do (
  if not defined PORT (
    "!PY!" -c "import socket,sys;s=socket.socket();s.settimeout(0.6);sys.exit(1 if s.connect_ex(('127.0.0.1',%%P))==0 else 0)" >nul 2>nul
    if !errorlevel!==0 set "PORT=%%P"
  )
)
if not defined PORT (
  if defined SILENT (
    echo [!TIME!] ERROR ports 5055/5056/5060/8080 all busy>>"!LOG!"
    exit /b 1
  )
  echo [错误] 5055 / 5056 / 5060 / 8080 都被占了，先关掉占用它们的程序再试。
  echo.
  pause
  exit /b 1
)

rem —— 3. 起服务 ——
if defined SILENT (
  echo [!TIME!] starting, port !PORT!>>"!LOG!"
  "!PY!" -u web/app.py >> "!RUNLOG!" 2>&1
  echo [!TIME!] server exited>>"!LOG!"
  exit /b 0
)

echo [3/3] 启动服务，端口 !PORT! ...
start "AutoResearch 服务 - 端口 !PORT!（关掉这个窗口=停止服务）" /min "!PY!" -u web/app.py

set "OK="
for /l %%i in (1,1,40) do (
  if not defined OK (
    "!PY!" -c "import urllib.request,sys;urllib.request.urlopen('http://127.0.0.1:!PORT!/api/health',timeout=1);sys.exit(0)" >nul 2>nul
    if !errorlevel!==0 (
      set "OK=1"
    ) else (
      ping -n 2 127.0.0.1 >nul
    )
  )
)

if defined OK (
  echo.
  echo 服务已经起来了：http://127.0.0.1:!PORT!/
  start "" "http://127.0.0.1:!PORT!/"
  echo 浏览器已打开。任务栏里那个最小化的「AutoResearch 服务」窗口就是后台，别关。
  ping -n 6 127.0.0.1 >nul
  exit /b 0
)

echo.
echo [错误] 等了 40 秒服务都没起来，手动跑一下看具体报错：
echo        "!PY!" -u web/app.py
echo.
pause
exit /b 1
