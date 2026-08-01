@echo off
rem -----------------------------------------------------------------------------
rem Step 4: Describe page layout/structure from PNGs (image-grounded, local
rem Gemma 3).
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
title Step 4 - Layout Description with Gemma 3

call "%SCRIPT_DIR%uv_setup.bat"
if errorlevel 1 (
	echo.
	echo UV setup failed. Exiting.
	pause
	goto :end
)

call "%SCRIPT_DIR%scripts\describe_layout_gemma.bat"

:end
endlocal
