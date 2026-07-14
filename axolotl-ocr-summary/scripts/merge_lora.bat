@echo off
REM Merges a trained LoRA/QLoRA adapter into a standalone full model.
REM Usage: scripts\merge_lora.bat [config-name] [adapter-dir]
REM adapter-dir defaults to output\<config-basename>\ (Axolotl's default output_dir)

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
uv run axolotl merge-lora "configs\%CONFIG%" --lora-model-dir="%ADAPTER_DIR%"
popd
