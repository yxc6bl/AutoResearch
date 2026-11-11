' 开机自启用的隐藏启动器：不起任何窗口，后台把网页服务拉起来。
' 真正的逻辑在 _svc_launch.bat -> 启动网页.bat service，这里只负责隐藏窗口。
Option Explicit
Dim sh, fso, base
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
base = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = base
sh.Run "cmd /c call """ & base & "\_svc_launch.bat""", 0, False
