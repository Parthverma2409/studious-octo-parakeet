@echo off
REM One-click demo launcher for Windows. Double-click this file.
REM Creates the virtual environment on first run, installs the project,
REM generates the NLU data the demo needs, then runs the narrated walkthrough.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.11 or newer from https://python.org
    echo and tick "Add python to PATH" during installation.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 ( echo Failed to create .venv & pause & exit /b 1 )
) else (
    echo [1/4] Virtual environment already exists.
)

set PY=.venv\Scripts\python.exe

echo [2/4] Installing project (first run only takes a minute)...
%PY% -m pip install --quiet --upgrade pip
%PY% -m pip install --quiet -e ".[dev]"
if errorlevel 1 ( echo Install failed & pause & exit /b 1 )

if not exist "data\nlu\splits\train.jsonl" (
    echo [3/4] Generating NLU dataset splits - needed by demo section 3...
    %PY% -m packages.nlp build >nul
    if errorlevel 1 ( echo NLU build failed & pause & exit /b 1 )
) else (
    echo [3/4] NLU dataset splits already present.
)

echo [4/4] Running the demo...
echo.
%PY% -m packages.demo %*

echo.
echo ---------------------------------------------------------------
echo Demo finished. To run one section only:  run_demo.bat --only 4
echo To list sections:                        run_demo.bat --list
echo Full setup and other commands:           SETUP.md
echo ---------------------------------------------------------------
pause
