@echo off
REM Runs eval\evaluate.py against the held-out validation split.
REM Usage: scripts\evaluate.bat [config-name] [adapter-dir]

setlocal
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=qlora-3090-24gb.yml
set ADAPTER_DIR=%2

set ROOT=%~dp0..
set HF_HOME=%ROOT%\.cache\huggingface

if "%ADAPTER_DIR%"=="" (
    for %%F in ("%CONFIG%") do set BASE=%%~nF
    set ADAPTER_DIR=%ROOT%\output\%BASE%
)

pushd "%ROOT%"
uv run python eval\evaluate.py --config "configs\%CONFIG%" --adapter-dir "%ADAPTER_DIR%"
popd
