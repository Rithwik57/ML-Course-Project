@echo off
setlocal

set "PY=C:\Users\RITHWIK\mambaforge\envs\gisrisk\python.exe"

if not exist "%PY%" (
  echo [ERROR] Could not find gisrisk Python at:
  echo         %PY%
  echo Update PY path in run_backend.bat and try again.
  exit /b 1
)

cd /d "%~dp0"
echo [INFO] Starting backend on http://127.0.0.1:8000
echo [INFO] First startup can take ~10-30 seconds while GIS data loads.
"%PY%" -m uvicorn src.api:app --host 127.0.0.1 --port 8000
