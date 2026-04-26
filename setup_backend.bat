@echo off
setlocal

set "PY=C:\Users\RITHWIK\mambaforge\envs\gisrisk\python.exe"

if not exist "%PY%" (
  echo [ERROR] Could not find gisrisk Python at:
  echo         %PY%
  echo Update PY path in setup_backend.bat and try again.
  exit /b 1
)

cd /d "%~dp0"
echo [INFO] Installing backend dependencies into gisrisk...
"%PY%" -m pip install -r requirements.txt

if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  exit /b 1
)

echo [OK] Backend dependencies are installed.
