@echo off
rem -----------------------------------------------------------------------------
rem Step 1: Convert PDF dataset to PNG pages.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
title Step 1 - Convert PDFs to PNGs

call "%SCRIPT_DIR%uv_setup.bat"
if errorlevel 1 (
	echo.
	echo UV setup failed. Exiting.
	pause
	goto :end
)

call "%SCRIPT_DIR%scripts\convert_pdf_to_png.bat"

:end
endlocal
