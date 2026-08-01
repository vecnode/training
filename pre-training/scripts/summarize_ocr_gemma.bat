@echo off
rem -----------------------------------------------------------------------------
rem Summarize OCR CSV rows with a local Gemma 3 model and write
rem [dataset-folder]-SUMMARIES.csv alongside the OCR CSV.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions EnableDelayedExpansion

rem Resolve paths.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"

title OCR CSV Summaries with Gemma 3
echo.
echo ================================================
echo   OCR CSV Summaries (local Gemma 3)
echo ================================================
echo.
echo Enter dataset base name, PNG/output folder path, or a direct OCR CSV path.
echo Examples: Release_1  or  C:\data\Release_1_PNG  or  C:\...\dataset-OCR.csv
echo.

set /p "DATASET_INPUT=Dataset: "
if "%DATASET_INPUT%"=="" (
	echo.
	echo No dataset input provided. Exiting.
	goto :end
)

set "OCR_FILE="
set "OUT_FILE="

rem A direct path to an OCR CSV is used as-is, bypassing folder resolution.
if /I "%DATASET_INPUT:~-4%"==".csv" if exist "%DATASET_INPUT%" set "OCR_FILE=%DATASET_INPUT%"

if defined OCR_FILE (
	for %%I in ("%OCR_FILE%") do (
		set "OCR_DIR=%%~dpI"
		set "OCR_BASENAME=%%~nI"
	)
	set "SUMMARY_BASENAME=!OCR_BASENAME!"
	if /I "!SUMMARY_BASENAME:~-4!"=="-OCR" set "SUMMARY_BASENAME=!SUMMARY_BASENAME:~0,-4!"
	set "OUT_FILE=!OCR_DIR!!SUMMARY_BASENAME!-SUMMARIES.csv"
) else (
	set "PNG_DIR="

	rem Resolve dataset input to the run folder: exact path, path under the
	rem project root, legacy "<name>_PNG" convention, or newest matching
	rem outputs\<timestamp>_<slug> folder from convert_pdf_to_png.ps1.
	for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%resolve_png_dir.ps1" -DatasetInput "%DATASET_INPUT%" -ProjectRoot "%ROOT_DIR%"`) do set "PNG_DIR=%%P"

	if not exist "!PNG_DIR!" (
		echo.
		echo Dataset folder not found for input: "%DATASET_INPUT%"
		echo Looked for: an exact path, a path under the project root, the legacy
		echo "<name>_PNG" convention, and outputs\[timestamp]_[slug] folders.
		goto :end
	)

	for %%I in ("!PNG_DIR!") do set "DATASET_FOLDER=%%~nxI"
	set "OCR_FILE=!PNG_DIR!\!DATASET_FOLDER!-OCR.csv"
	set "OUT_FILE=!PNG_DIR!\!DATASET_FOLDER!-SUMMARIES.csv"

	if not exist "!OCR_FILE!" (
		echo.
		echo OCR CSV not found: "!OCR_FILE!"
		echo Run exec_2.bat / ocr_detection_png.bat on this folder first.
		goto :end
	)
)

if exist "!OUT_FILE!" (
	set "OCR_LINES=0"
	set "SUM_LINES=0"
	for /f %%N in ('type "!OCR_FILE!" 2^>nul ^| find /c /v ""') do set "OCR_LINES=%%N"
	for /f %%N in ('type "!OUT_FILE!" 2^>nul ^| find /c /v ""') do set "SUM_LINES=%%N"
	set /a "OCR_ROWS=OCR_LINES-1"
	set /a "DONE_COUNT=SUM_LINES-1"
	if !OCR_ROWS! LSS 0 set "OCR_ROWS=0"
	if !DONE_COUNT! LSS 0 set "DONE_COUNT=0"
	echo.
	echo Found existing summaries: !DONE_COUNT! / !OCR_ROWS! page^(s^) already processed in:
	echo   !OUT_FILE!
	echo This run resumes automatically and only summarizes the remaining pages.
)

echo.
set /p "MODEL_ID=Model id (blank = unsloth/gemma-3-4b-it, ungated): "

echo.
echo Input : !OCR_FILE!
echo Output: !OUT_FILE!
echo.

rem Execute summarization, optionally with an explicit model.
if defined MODEL_ID (
	"%UFO_PYTHON%" "%SCRIPT_DIR%summarize_ocr_gemma.py" --input "!OCR_FILE!" --output "!OUT_FILE!" --model-id "%MODEL_ID%"
) else (
	"%UFO_PYTHON%" "%SCRIPT_DIR%summarize_ocr_gemma.py" --input "!OCR_FILE!" --output "!OUT_FILE!"
)

echo.
echo Summary run finished.

:end
echo.
pause
