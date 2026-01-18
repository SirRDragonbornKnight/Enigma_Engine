' ForgeAI Silent Launcher for Windows
' Double-click this to start ForgeAI without a terminal window
'
' This script launches ForgeAI using pythonw.exe which runs without a console.
' If you need to see error output, use ForgeAI.bat instead.

Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Get the directory where this script is located
strScriptPath = FSO.GetParentFolderName(WScript.ScriptFullName)

' Change to the script directory
WshShell.CurrentDirectory = strScriptPath

' Launch with pythonw (no terminal window)
' The 0 means window is hidden, False means don't wait for it to finish
WshShell.Run "pythonw run.py --gui", 0, False

Set WshShell = Nothing
Set FSO = Nothing
