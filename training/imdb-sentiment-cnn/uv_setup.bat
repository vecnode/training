@echo off
rem -----------------------------------------------------------------------------
rem Create/sync local uv environment and validate CUDA-ready torch.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

rem Resolve absolute project root path.
pushd "%~dp0" >nul
set "SCRIPT_DIR=%CD%"
popd >nul

set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"

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

rem Sync project dependencies from lock file. --inexact is deliberate: without
rem it, `uv sync` also prunes torch itself despite the --no-install-package
rem flag, undoing the CUDA wheel install below.
echo.
echo Checking local dependencies with uv...
uv sync --project "%SCRIPT_DIR%" --python "%VENV_PY%" --frozen --inexact --no-install-package torch --check >nul 2>nul
if errorlevel 1 (
    echo Syncing local dependencies with uv...
    uv sync --project "%SCRIPT_DIR%" --python "%VENV_PY%" --frozen --inexact --no-install-package torch
    if errorlevel 1 exit /b 1
) else (
    echo Local dependencies are already synced. Skipping uv sync.
)

rem Require NVIDIA tooling for GPU-first mode.
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo.
    echo NVIDIA GPU was not detected via nvidia-smi.
    echo GPU-first mode requires an NVIDIA driver and CUDA-capable GPU.
    exit /b 1
)

rem Install CUDA wheel when torch is missing or CPU-only.
echo.
set "NEED_TORCH_INSTALL=1"
"%VENV_PY%" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
if not errorlevel 1 (
    "%VENV_PY%" -c "import sys,torch; sys.exit(0 if torch.version.cuda else 1)"
    if not errorlevel 1 set "NEED_TORCH_INSTALL=0"
)

if "%NEED_TORCH_INSTALL%"=="1" (
    echo Installing CUDA-enabled PyTorch wheel...
    uv pip install --python "%VENV_PY%" --index-url "%TORCH_INDEX_URL%" --reinstall torch
    if errorlevel 1 exit /b 1
) else (
    echo CUDA-enabled torch already installed. Skipping install.
)

rem Final CUDA sanity check in the local venv.
echo.
echo Verifying CUDA in local environment...
"%VENV_PY%" -c "import sys, torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo.
    echo CUDA validation failed in .venv. GPU execution would fall back to CPU.
    echo Check NVIDIA driver compatibility and rerun uv_setup.bat.
    exit /b 1
)

endlocal & set "IMDB_SENTIMENT_CNN_PYTHON=%VENV_PY%" & exit /b 0
