@echo off
REM UAP Analysis Suite - Quick Build Script for Windows

setlocal enabledelayedexpansion

REM Colors for output (using echo with color codes)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo %BLUE%🛸 UAP Analysis Suite - Build Script%NC%
echo ==================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR] Python is not installed or not in PATH%NC%
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %BLUE%[INFO] Using Python %PYTHON_VERSION%%NC%

REM Check minimum Python version (3.8)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR] Python 3.8 or higher is required%NC%
    pause
    exit /b 1
)

REM Parse command line arguments
set CLEAN_BUILD=false
set VERBOSE=false
set PLATFORMS=current
set INSTALL_DEPS=true

:parse_args
if "%1"=="" goto :end_parse_args
if "%1"=="--clean" (
    set CLEAN_BUILD=true
    shift
    goto :parse_args
)
if "%1"=="--verbose" (
    set VERBOSE=true
    shift
    goto :parse_args
)
if "%1"=="--platforms" (
    set PLATFORMS=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--no-deps" (
    set INSTALL_DEPS=false
    shift
    goto :parse_args
)
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="/?" goto :show_help

echo %YELLOW%[WARNING] Unknown option: %1%NC%
shift
goto :parse_args

:show_help
echo UAP Analysis Suite Build Script
echo.
echo Usage: %0 [OPTIONS]
echo.
echo Options:
echo   --clean      Clean previous build artifacts
echo   --verbose    Enable verbose output
echo   --platforms  Platforms to build for (current,linux,windows,macos,all)
echo   --no-deps    Skip dependency installation
echo   --help, -h, /?   Show this help message
echo.
echo Examples:
echo   %0                           # Quick build for current platform
echo   %0 --clean --verbose        # Clean build with verbose output
echo   %0 --platforms all          # Build for all platforms
echo   %0 --no-deps               # Build without installing dependencies
echo.
pause
exit /b 0

:end_parse_args

REM Install dependencies
if "%INSTALL_DEPS%"=="true" (
    echo %BLUE%[INFO] Installing build dependencies...%NC%
    
    if exist "requirements.txt" (
        python -m pip install -r requirements.txt
        if errorlevel 1 (
            echo %RED%[ERROR] Failed to install requirements.txt%NC%
            pause
            exit /b 1
        )
    )
    
    if exist "build_requirements.txt" (
        python -m pip install -r build_requirements.txt
        if errorlevel 1 (
            echo %RED%[ERROR] Failed to install build_requirements.txt%NC%
            pause
            exit /b 1
        )
    )
    
    echo %GREEN%[SUCCESS] Dependencies installed%NC%
)

REM Build the application
echo %BLUE%[INFO] Starting UAP Analysis Suite build...%NC%
echo %BLUE%[INFO] Build configuration:%NC%
echo %BLUE%[INFO]   Platforms: %PLATFORMS%%NC%
echo %BLUE%[INFO]   Clean build: %CLEAN_BUILD%%NC%
echo %BLUE%[INFO]   Verbose: %VERBOSE%%NC%

REM Prepare build arguments
set BUILD_ARGS=
if "%CLEAN_BUILD%"=="true" (
    set BUILD_ARGS=!BUILD_ARGS! --clean
)
if "%VERBOSE%"=="true" (
    set BUILD_ARGS=!BUILD_ARGS! --verbose
)
if not "%PLATFORMS%"=="current" (
    set BUILD_ARGS=!BUILD_ARGS! --platforms %PLATFORMS%
)

REM Run the build
echo %BLUE%[INFO] Running: python build_all.py%BUILD_ARGS%%NC%
python build_all.py %BUILD_ARGS%

if errorlevel 1 (
    echo %RED%[ERROR] Build failed!%NC%
    pause
    exit /b 1
)

echo %GREEN%[SUCCESS] Build completed successfully!%NC%
echo %BLUE%[INFO] Check build-compile-dist\packages\ for output files%NC%

REM Check if packages were created
if exist "build-compile-dist\packages" (
    echo.
    echo %GREEN%📦 Generated packages:%NC%
    dir /b "build-compile-dist\packages\*.*" 2>nul | findstr /v /c:"build_report" | head -5
    echo.
)

echo %GREEN%All done! 🚀%NC%
pause
exit /b 0