@echo off
cd /d "%~dp0"
title Publish to GitHub

where git >nul 2>nul
if errorlevel 1 (
  echo Git was not found. Install it from https://git-scm.com then double-click this file again.
  pause
  exit /b
)

echo 1) On github.com create a NEW EMPTY repository (do NOT add a README).
echo 2) Copy its address, for example https://github.com/your-name/your-repo.git
echo.
set /p REPO=Paste the repository address here: 

git init
for /f "delims=" %%i in (
'git config user.email'
) do set EMAIL=%%i
if "%EMAIL%"=="" goto askname
goto commit
:askname
set /p NAME=Your name: 
set /p EMAIL=Your email: 
git config user.name "%NAME%"
git config user.email "%EMAIL%"
:commit
git add .
git commit -m "Customer Intelligence System"
git branch -M main
git remote remove origin >nul 2>nul
git remote add origin %REPO%
git push -u origin main --force
echo.
echo If a browser window asks you to sign in to GitHub, approve it, then run this file again if the push did not finish.
pause
