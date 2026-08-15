@echo off
rem -----------------------------------------------------------------------------
rem Deploy the trained Vicuna-7B LoRA adapter inference server (FastAPI + uvicorn).
rem Bootstraps this folder's own local uv environment, then serves app.py.
rem
rem Usage:  deploy.bat [host] [port]      (defaults: 127.0.0.1 8008)
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

rem Resolve this folder's absolute path (self-contained: own venv, own deps).
pushd "%~dp0" >nul
set "DEPLOY_DIR=%CD%"
popd >nul

set "VENV_PY=%DEPLOY_DIR%\.venv\Scripts\python.exe"

rem Optional host/port overrides.
set "APP_HOST=%~1"
set "APP_PORT=%~2"
if "%APP_HOST%"=="" set "APP_HOST=127.0.0.1"
if "%APP_PORT%"=="" set "APP_PORT=8008"

title Vicuna-7B LoRA Adapter Inference Server

rem Ensure the local uv environment + CUDA-ready torch are present.
call "%DEPLOY_DIR%\uv_setup.bat"
if errorlevel 1 (
    echo.
    echo UV setup failed. Cannot start the inference server.
    exit /b 1
)

if not exist "%VENV_PY%" (
    echo.
    echo Local venv python not found at "%VENV_PY%".
    echo Run uv_setup.bat from this folder first.
    exit /b 1
)

echo.
echo Starting adapter inference server on http://%APP_HOST%:%APP_PORT%
echo   model source: merged_model\ (if present) else ..\..\vicuna-7b-lora\runs\ adapter
echo   front-end: http://%APP_HOST%:%APP_PORT%/   API: /api/summarize, /api/health
echo.
"%VENV_PY%" "%DEPLOY_DIR%\app.py" --host %APP_HOST% --port %APP_PORT%

endlocal
