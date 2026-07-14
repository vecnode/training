@echo off
REM Quick manual prompt test against a trained adapter.
REM Usage: scripts\inference.bat [config-name] [adapter-dir]

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
uv run axolotl inference "configs\%CONFIG%" --lora-model-dir="%ADAPTER_DIR%"
popd
