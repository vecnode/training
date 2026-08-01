@echo off
rem -----------------------------------------------------------------------------
rem Step 3: Summarize OCR text with a local Gemma 3 model (no Ollama).
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
title Step 3 - Summarize OCR with Gemma 3

call "%SCRIPT_DIR%uv_setup.bat"
if errorlevel 1 (
	echo.
	echo UV setup failed. Exiting.
	pause
	goto :end
)

call "%SCRIPT_DIR%scripts\summarize_ocr_gemma.bat"

:end
endlocal
