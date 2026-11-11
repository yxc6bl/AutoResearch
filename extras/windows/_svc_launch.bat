@echo off
rem 开机自启的中间层：把调用转给 启动网页.bat 的 service 模式。
rem 单独拆出来是为了让 _svc_launch.vbs 里不出现中文，避免脚本编码问题。
cd /d "%~dp0"
call "启动网页.bat" service
