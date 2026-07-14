@echo off
REM Usage: scripts\train.bat [config-name]
REM Default config-name is qlora-3090-24gb.yml

setlocal
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=qlora-3090-24gb.yml

set ROOT=%~dp0..
set HF_HOME=%ROOT%\.cache\huggingface

pushd "%ROOT%"
uv run axolotl train "configs\%CONFIG%"
popd
