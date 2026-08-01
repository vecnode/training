@echo off
rem -----------------------------------------------------------------------------
rem Step 2: OCR PNG pages with Surya OCR.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
title Step 2 - OCR PNGs

call "%SCRIPT_DIR%uv_setup.bat"
if errorlevel 1 (
	echo.
	echo UV setup failed. Exiting.
	pause
	goto :end
)

call "%SCRIPT_DIR%scripts\ocr_detection_png.bat"

:end
endlocal
