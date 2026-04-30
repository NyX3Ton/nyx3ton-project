@echo off
cd /d %~dp0
echo Starting Local AI CV Validator...
python app-zero_shot.py
pause
