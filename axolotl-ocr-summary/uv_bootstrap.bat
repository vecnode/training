@echo off
REM Bootstraps the uv-managed environment for this workspace.
REM Every download (venv, python, pip cache, HF model weights, HF datasets)
REM is redirected under .cache\ and .venv\ in this repo, both gitignored.
REM `rmdir /s /q .venv .cache` fully resets the workspace.

setlocal
set ROOT=%~dp0
set UV_CACHE_DIR=%ROOT%.cache\uv
set HF_HOME=%ROOT%.cache\huggingface
set HF_HUB_ENABLE_HF_TRANSFER=1

if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found. Install it first: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

pushd "%ROOT%"
uv venv --python 3.11
uv sync

REM axolotl pulls in flash-attn / xformers style extras that need torch already
REM present and built without isolation — reinstall it explicitly after sync.
uv pip install --no-build-isolation "axolotl[deepspeed]"

uv run axolotl fetch examples --dest .cache\axolotl-examples
uv run axolotl fetch deepspeed_configs --dest .cache\deepspeed_configs
popd

echo Bootstrap complete. Activate with: .venv\Scripts\activate.bat
