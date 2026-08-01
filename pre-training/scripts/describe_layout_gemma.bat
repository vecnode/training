@echo off
rem -----------------------------------------------------------------------------
rem Describe page layout/structure from PNG pages (image-grounded, local
rem Gemma 3) and write [dataset-folder]-LAYOUT.csv alongside them.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions EnableDelayedExpansion

rem Resolve paths.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"

title Layout Description on PNG Dataset
echo.
echo ================================================
echo   Layout Description (PNG -^> [dataset-folder]-LAYOUT.csv)
echo ================================================
echo.
echo Enter dataset base name OR PNG folder path.
echo Examples: Release_1  or  Release_1_PNG  or  C:\data\Release_1_PNG
echo.

set /p "DATASET_INPUT=Dataset: "
if "%DATASET_INPUT%"=="" (
	echo.
	echo No dataset input provided. Exiting.
	goto :end
)

set "PNG_DIR="

rem Resolve dataset input to a PNG folder path: exact path, path under the
rem project root, legacy "<name>_PNG" convention, or newest matching
rem outputs\<timestamp>_<slug> folder from convert_pdf_to_png.ps1.
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%resolve_png_dir.ps1" -DatasetInput "%DATASET_INPUT%" -ProjectRoot "%ROOT_DIR%"`) do set "PNG_DIR=%%P"

if not exist "%PNG_DIR%" (
	echo.
	echo PNG dataset folder not found for input: "%DATASET_INPUT%"
	echo Looked for: an exact path, a path under the project root, the legacy
	echo "<name>_PNG" convention, and outputs\[timestamp]_[slug] folders.
	echo Run convert_pdf_to_png.bat / exec_1.bat first.
	goto :end
)

rem The LAYOUT CSV lives alongside the PNGs it describes, named after the
rem dataset folder itself. Re-runs against the same folder resume the same
rem file instead of colliding with a different run of a similarly-named
rem dataset.
for %%I in ("%PNG_DIR%") do set "DATASET_FOLDER=%%~nxI"
set "OUT_FILE=%PNG_DIR%\%DATASET_FOLDER%-LAYOUT.csv"

if exist "%OUT_FILE%" (
	set "PNG_COUNT=0"
	set "CSV_LINES=0"
	for /f %%N in ('dir /b "%PNG_DIR%\*.png" 2^>nul ^| find /c /v ""') do set "PNG_COUNT=%%N"
	for /f %%N in ('type "%OUT_FILE%" 2^>nul ^| find /c /v ""') do set "CSV_LINES=%%N"
	set /a "DONE_COUNT=CSV_LINES-1"
	if !DONE_COUNT! LSS 0 set "DONE_COUNT=0"
	echo.
	echo Found existing layout descriptions: !DONE_COUNT! / !PNG_COUNT! page^(s^) already processed in:
	echo   !OUT_FILE!
	echo This run resumes automatically and only describes the remaining pages.
)

echo.
echo Input : %PNG_DIR%
echo Output: %OUT_FILE%
echo.

rem Execute layout description pipeline.
"%UFO_PYTHON%" "%SCRIPT_DIR%describe_layout_gemma.py" --image-dir "%PNG_DIR%" --output "%OUT_FILE%"

echo.
echo Layout description run finished.

:end
echo.
pause
