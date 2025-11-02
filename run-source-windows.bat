@echo off
REM Run UAP Analysis from source on Windows

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    pause
    exit /b 1
)

REM Check for virtual environment
if exist "%PROJECT_DIR%\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%PROJECT_DIR%\venv\Scripts\activate.bat"
) else if exist "%PROJECT_DIR%\.venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
)

REM Set Python path
set PYTHONPATH=%PROJECT_DIR%\src;%PYTHONPATH%

REM Run the application
echo Starting UAP Analysis GUI...
python -m gui.stable_gui

pause