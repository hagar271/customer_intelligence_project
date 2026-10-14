@echo off
cd /d "%~dp0"
title Build results

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add Python to PATH", then double-click this file again.
  pause
  exit /b
)

if not exist venv (
  echo Creating a virtual environment... first time only
  python -m venv venv
)
call venv\Scripts\activate
echo Installing requirements... first time can take a few minutes
python -m pip install -r requirements.txt

echo.
echo Building the results (a file window may ask you to choose online_retail_II.xlsx)...
python build_results.py
echo.
pause
