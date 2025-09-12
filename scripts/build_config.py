#!/usr/bin/env python3
"""
Cross-Platform Build Configuration for UAP Analysis Suite
Unified PyInstaller configuration for macOS, Windows, and Linux builds
"""

import os
import sys
import platform
from pathlib import Path

# Build metadata
APP_NAME = "UAP Video Analyzer"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Advanced Scientific Analysis Tool for Unidentified Aerial Phenomena"
BUNDLE_ID = "org.uapanalysis.videoanalyzer"
AUTHOR = "UAP Analysis Team"
COPYRIGHT = f"Copyright © 2024 {AUTHOR}"

# Paths
PROJECT_DIR = Path(__file__).parent
SRC_DIR = PROJECT_DIR / "src"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
ASSETS_DIR = PROJECT_DIR / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
BUILD_DIR = PROJECT_DIR / "build"
DIST_DIR = PROJECT_DIR / "dist"

# Main entry points
MAIN_SCRIPT = SCRIPTS_DIR / "uap_gui.py"
CONSOLE_SCRIPT = SCRIPTS_DIR / "run_analysis.py"

# Platform detection
CURRENT_OS = platform.system().lower()
IS_MACOS = CURRENT_OS == "darwin"
IS_WINDOWS = CURRENT_OS == "windows"
IS_LINUX = CURRENT_OS == "linux"

# PyInstaller base configuration
BASE_CONFIG = {
    'name': APP_NAME.replace(" ", "_"),
    'pathex': [str(PROJECT_DIR), str(SRC_DIR)],
    'binaries': [],
    'datas': [
        (str(SRC_DIR), 'src'),
        (str(PROJECT_DIR / 'configs'), 'configs'),
        (str(PROJECT_DIR / 'data'), 'data'),
        (str(ASSETS_DIR), 'assets'),
    ],
    'hiddenimports': [
        # Core modules
        'src.analyzers',
        'src.processors', 
        'src.visualizers',
        'src.utils',
        # Scientific computing
        'numpy',
        'scipy',
        'matplotlib',
        'cv2',
        'PIL',
        'sklearn',
        # GUI frameworks
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        # PyTorch and ML
        'torch',
        'torchvision',
        'transformers',
        # Additional dependencies
        'yaml',
        'tqdm',
        'open3d',
        'scikit-image',
        # Platform-specific
        'threading',
        'subprocess',
        'multiprocessing',
    ],
    'hookspath': [],
    'runtime_hooks': [],
    'excludes': [
        # Remove unused modules to reduce size
        'matplotlib.tests',
        'numpy.tests',
        'scipy.tests',
        'PIL.tests',
        'tkinter.test',
        'test',
        'tests',
        'pytest',
        'doctest',
        'unittest',
    ],
    'win_no_prefer_redirects': False,
    'win_private_assemblies': False,
    'cipher': None,
    'noarchive': False,
    'optimize': 0,
}

# Platform-specific configurations
def get_macos_config():
    """macOS-specific build configuration"""
    icon_path = ICONS_DIR / 'macos' / 'app_icon.icns'
    
    config = BASE_CONFIG.copy()
    config.update({
        'icon': str(icon_path) if icon_path.exists() else None,
        'console': False,  # GUI app
        'target_arch': None,  # Use native architecture to avoid fat binary issues
        'codesign_identity': None,  # Set to signing identity for distribution
        'entitlements_file': None,  # Add entitlements.plist for sandboxing
    })
    
    # macOS bundle info
    config['bundle_identifier'] = BUNDLE_ID
    config['info_plist'] = {
        'CFBundleName': APP_NAME,
        'CFBundleDisplayName': APP_NAME,
        'CFBundleIdentifier': BUNDLE_ID,
        'CFBundleVersion': APP_VERSION,
        'CFBundleShortVersionString': APP_VERSION,
        'CFBundleInfoDictionaryVersion': '6.0',
        'NSHumanReadableCopyright': COPYRIGHT,
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13.0',
        'NSRequiresAquaSystemAppearance': False,
        'NSCameraUsageDescription': 'UAP Analysis requires camera access for live video analysis',
        'NSMicrophoneUsageDescription': 'UAP Analysis requires microphone access for acoustic analysis',
        'NSDocumentsFolderUsageDescription': 'UAP Analysis needs access to save analysis results',
    }
    
    return config

def get_windows_config():
    """Windows-specific build configuration"""
    icon_path = ICONS_DIR / 'windows' / 'app_icon.ico'
    
    config = BASE_CONFIG.copy()
    config.update({
        'icon': str(icon_path) if icon_path.exists() else None,
        'console': False,  # GUI app (set to True for console debugging)
        'uac_admin': False,  # Don't require admin privileges
        'uac_uiaccess': False,
    })
    
    # Windows version info
    config['version_file'] = {
        'version': APP_VERSION,
        'description': APP_DESCRIPTION,
        'company': AUTHOR,
        'product': APP_NAME,
        'copyright': COPYRIGHT,
        'trademarks': '',
        'file_description': APP_DESCRIPTION,
        'internal_name': APP_NAME.replace(" ", ""),
        'original_filename': f"{APP_NAME.replace(' ', '')}.exe",
    }
    
    return config

def get_linux_config():
    """Linux-specific build configuration"""
    icon_path = ICONS_DIR / 'linux' / 'app_icon_256x256.png'
    
    config = BASE_CONFIG.copy()
    config.update({
        'icon': str(icon_path) if icon_path.exists() else None,
        'console': False,  # GUI app
    })
    
    return config

# Console application configurations (for command-line tools)
def get_console_config():
    """Configuration for console-based analysis tool"""
    config = BASE_CONFIG.copy()
    config.update({
        'name': 'uap_analyzer_cli',
        'console': True,
        'icon': None,  # Console apps don't need icons
    })
    
    return config

# Export configurations based on current platform
def get_current_platform_config():
    """Get configuration for current platform"""
    if IS_MACOS:
        return get_macos_config()
    elif IS_WINDOWS:
        return get_windows_config()
    elif IS_LINUX:
        return get_linux_config()
    else:
        raise RuntimeError(f"Unsupported platform: {CURRENT_OS}")

# Build output directories
BUILD_DIRS = {
    'macos': DIST_DIR / 'macos',
    'darwin': DIST_DIR / 'macos',  # macOS alias for platform.system()
    'windows': DIST_DIR / 'windows', 
    'linux': DIST_DIR / 'linux',
    'dist': DIST_DIR,
    'packages': DIST_DIR / 'packages',
    'build': BUILD_DIR,  # Temporary build artifacts
}

# Create build directories
for build_dir in BUILD_DIRS.values():
    build_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"🏗️  Build Configuration for UAP Analysis Suite")
    print(f"Platform: {CURRENT_OS}")
    print(f"Version: {APP_VERSION}")
    print(f"Main Script: {MAIN_SCRIPT}")
    
    config = get_current_platform_config()
    print(f"Icon: {config.get('icon', 'None')}")
    print(f"Console Mode: {config.get('console', False)}")
    print(f"Build Directory: {BUILD_DIR}")
    
    print("\n✅ Configuration loaded successfully!")