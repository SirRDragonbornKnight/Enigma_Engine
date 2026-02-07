' Enigma AI Engine Silent Launcher for Windows
' Double-click this to start Enigma AI Engine without a terminal window
'
' This script launches Enigma AI Engine using pythonw.exe which runs without a console.
' If you need to see error output, use Enigma AI Engine.bat instead.

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
