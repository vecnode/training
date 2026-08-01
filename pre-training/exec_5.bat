@echo off
rem -----------------------------------------------------------------------------
rem Step 5: Generate synthetic QA pairs from OCR text (local Gemma 3).
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
title Step 5 - Synthetic QA Pairs with Gemma 3

call "%SCRIPT_DIR%uv_setup.bat"
if errorlevel 1 (
	echo.
	echo UV setup failed. Exiting.
	pause
	goto :end
)

call "%SCRIPT_DIR%scripts\generate_qa_gemma.bat"

:end
endlocal
