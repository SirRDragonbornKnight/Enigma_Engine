@echo off
cd /d "C:\Users\sirkn_gbhnunq\Documents\GitHub\Forge_AI"
git add -A
git commit -m "Avatar: Add 3D rotation, loading overlay, quick switch"
git push
del "%~f0"
