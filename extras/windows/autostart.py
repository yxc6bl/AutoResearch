"""管理 AutoResearch 网页服务的开机自启（Windows「启动」文件夹里放一个静默启动脚本）。

用法：python _tools/autostart.py on|off|status
不写注册表、不需要管理员权限，改动只有「启动」文件夹里的一个 .vbs 文件。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTUP = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
ENTRY = STARTUP / "AutoResearch 服务.vbs"

TEMPLATE = '''\' AutoResearch 网页服务 · 登录时静默启动（不会弹任何窗口）
\' 由项目里的「开启开机自启.bat」生成；要取消就双击「关闭开机自启.bat」，或直接删掉本文件。
Option Explicit
Dim sh, fso, launcher, logPath
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

launcher = "{launcher}"
logPath = "{log}"
If Not fso.FileExists(launcher) Then
  On Error Resume Next
  Dim lg
  Set lg = fso.OpenTextFile(logPath, 8, True)
  If Err.Number = 0 Then
    lg.WriteLine "[" & Now & "] startup: launcher not found -> " & launcher
    lg.Close
  End If
  On Error GoTo 0
  WScript.Quit 1
End If

sh.CurrentDirectory = "{root}"
sh.Run "cmd /c call """ & launcher & """", 0, False
'''


def entry_text() -> str:
    return TEMPLATE.format(
        launcher=ROOT / "_svc_launch.bat",
        log=ROOT / "logs" / "service.log",
        root=ROOT,
    ).replace("\n", "\r\n")


def main() -> int:
    action = (sys.argv[1] if len(sys.argv) > 1 else "status").lower()
    if action == "on":
        STARTUP.mkdir(parents=True, exist_ok=True)
        (ROOT / "logs").mkdir(exist_ok=True)
        ENTRY.write_bytes(entry_text().encode("gbk"))
        print("已开启开机自启。")
        print(f"  入口文件：{ENTRY}")
        print(f"  启动脚本：{ROOT / '_svc_launch.bat'}")
        print("  下次登录 Windows 会静默起服务（5055），之后双击网页即可直接用。")
        return 0
    if action == "off":
        if ENTRY.exists():
            ENTRY.unlink()
            print(f"已关闭开机自启，删除：{ENTRY}")
        else:
            print(f"本来就没有开机自启项：{ENTRY}")
        print("  当前正在跑的服务不受影响；要停它双击「停止网页服务.bat」。")
        return 0
    if action == "status":
        if ENTRY.exists():
            print(f"开机自启：已开启 → {ENTRY}")
        else:
            print(f"开机自启：未开启（{ENTRY} 不存在）")
        return 0
    print("用法：python _tools/autostart.py on|off|status")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
