@echo off
rem -----------------------------------------------------------------------------
rem Main pipeline controller menu.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

rem Resolve repository root from this script location.
set "SCRIPT_DIR=%~dp0"
title Dataset Pre-Training Controller

call "%SCRIPT_DIR%uv_setup.bat"
if errorlevel 1 (
	echo.
	echo UV setup failed. Exiting.
	pause
	goto end
)

:menu
cls
echo.
echo ================================================
echo   Dataset Pre-Training - Main Controller
echo ================================================
echo.
echo Choose an option:
echo.
echo   [1] Convert PDFs to PNGs
echo   [2] OCR text from PNGs
echo   [3] UV setup only (create/sync local env)
echo   [4] Summarize OCR CSV with local Gemma 3
echo   [5] Describe page layout from PNGs with local Gemma 3
echo   [6] Generate synthetic QA pairs from OCR CSV with local Gemma 3
echo   [0] Exit
echo.

set /p "OPT=Selection: "

if "%OPT%"=="1" goto convert
if "%OPT%"=="2" goto ocr
if "%OPT%"=="3" goto bootstrap
if "%OPT%"=="4" goto summarize
if "%OPT%"=="5" goto layout
if "%OPT%"=="6" goto qa
if "%OPT%"=="0" goto end

echo.
echo Invalid option. Please choose 0, 1, 2, 3, 4, 5, or 6.
pause
goto menu

:convert
call "%SCRIPT_DIR%scripts\convert_pdf_to_png.bat"
goto menu

:ocr
call "%SCRIPT_DIR%scripts\ocr_detection_png.bat"
goto menu

:summarize
call "%SCRIPT_DIR%scripts\summarize_ocr_gemma.bat"
goto menu

:layout
call "%SCRIPT_DIR%scripts\describe_layout_gemma.bat"
goto menu

:qa
call "%SCRIPT_DIR%scripts\generate_qa_gemma.bat"
goto menu

:bootstrap
call "%SCRIPT_DIR%uv_setup.bat"
echo.
if errorlevel 1 (
	echo UV setup failed.
) else (
	echo UV setup completed.
)
pause
goto menu

:end
echo.
echo Exiting.
endlocal
