@echo off
cd /d "%~dp0"
title Customer Intelligence System

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add Python to PATH" during installation, then double-click this file again.
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
echo Starting the dashboard. Your browser will open at http://localhost:8501
echo Close this window to stop the app.
python -m streamlit run app.py
pause
