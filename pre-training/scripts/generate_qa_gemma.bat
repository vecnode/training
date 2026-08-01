@echo off
rem -----------------------------------------------------------------------------
rem Generate synthetic QA pairs from an OCR CSV with a local Gemma 3 model and
rem write [dataset-folder]-QA.csv alongside the OCR CSV.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions EnableDelayedExpansion

rem Resolve paths.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"

title OCR CSV Synthetic QA Pairs with Gemma 3
echo.
echo ================================================
echo   Synthetic QA Pairs (local Gemma 3)
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
	set "QA_BASENAME=!OCR_BASENAME!"
	if /I "!QA_BASENAME:~-4!"=="-OCR" set "QA_BASENAME=!QA_BASENAME:~0,-4!"
	set "OUT_FILE=!OCR_DIR!!QA_BASENAME!-QA.csv"
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
	set "OUT_FILE=!PNG_DIR!\!DATASET_FOLDER!-QA.csv"

	if not exist "!OCR_FILE!" (
		echo.
		echo OCR CSV not found: "!OCR_FILE!"
		echo Run exec_2.bat / ocr_detection_png.bat on this folder first.
		goto :end
	)
)

if exist "!OUT_FILE!" (
	set "QA_LINES=0"
	for /f %%N in ('type "!OUT_FILE!" 2^>nul ^| find /c /v ""') do set "QA_LINES=%%N"
	set /a "DONE_COUNT=QA_LINES-1"
	if !DONE_COUNT! LSS 0 set "DONE_COUNT=0"
	echo.
	echo Found existing QA pairs: !DONE_COUNT! row^(s^) already written to:
	echo   !OUT_FILE!
	echo This run resumes automatically and only processes pages not yet covered.
	echo ^(Note: each page can produce multiple QA rows, so this count is not
	echo directly comparable to a page count.^)
)

echo.
set /p "MODEL_ID=Model id (blank = unsloth/gemma-3-4b-it, ungated): "

echo.
echo Input : !OCR_FILE!
echo Output: !OUT_FILE!
echo.

rem Execute QA generation, optionally with an explicit model.
if defined MODEL_ID (
	"%UFO_PYTHON%" "%SCRIPT_DIR%generate_qa_gemma.py" --input "!OCR_FILE!" --output "!OUT_FILE!" --model-id "%MODEL_ID%"
) else (
	"%UFO_PYTHON%" "%SCRIPT_DIR%generate_qa_gemma.py" --input "!OCR_FILE!" --output "!OUT_FILE!"
)

echo.
echo QA generation run finished.

:end
echo.
pause
