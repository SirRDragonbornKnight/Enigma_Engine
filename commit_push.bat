@echo off
REM Navigate to project directory (assumes script is in project root)
cd /d "%~dp0"
git add -A
git commit -m "Update"
git push
