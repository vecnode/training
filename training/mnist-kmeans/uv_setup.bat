@echo off
rem -----------------------------------------------------------------------------
rem Create/sync local uv environment. No CUDA/torch here - this pipeline only
rem depends on numpy, so setup is a plain venv + sync.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

rem Resolve absolute project root path.
pushd "%~dp0" >nul
set "SCRIPT_DIR=%CD%"
popd >nul

set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

rem Ensure uv exists before continuing.
where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo uv is required but was not found in PATH.
    echo Install uv from: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

if not exist "%VENV_PY%" (
    echo.
    echo Creating local virtual environment with uv...
    uv venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)

echo.
echo Checking local dependencies with uv...
uv sync --project "%SCRIPT_DIR%" --python "%VENV_PY%" --frozen --check >nul 2>nul
if errorlevel 1 (
    echo Syncing local dependencies with uv...
    uv sync --project "%SCRIPT_DIR%" --python "%VENV_PY%"
    if errorlevel 1 exit /b 1
) else (
    echo Local dependencies are already synced. Skipping uv sync.
)

endlocal & set "MNIST_KMEANS_PYTHON=%VENV_PY%" & exit /b 0
