#!/usr/bin/env python3
"""
UAP Analysis Suite - Master Build Script
========================================

Orchestrates the complete build-compile-dist process for all platforms.
Creates standalone executables, installers, and distribution packages.
"""

import os
import sys
import shutil
import subprocess
import platform
import argparse
from pathlib import Path
import tempfile
import json
from datetime import datetime

# Import our build configuration
from build_config import (
    get_current_platform_config, 
    get_console_config,
    BUILD_DIRS,
    APP_NAME,
    APP_VERSION,
    PROJECT_DIR,
    MAIN_SCRIPT,
    CONSOLE_SCRIPT
)

class UAP_Builder:
    def __init__(self, clean=False, verbose=False):
        self.clean = clean
        self.verbose = verbose
        self.platform = platform.system().lower()
        
        # Build paths
        self.platform_dir = BUILD_DIRS[self.platform]  # Final platform output in /dist
        self.build_dir = BUILD_DIRS['build']  # Temporary build artifacts
        self.dist_dir = BUILD_DIRS['dist']
        self.packages_dir = BUILD_DIRS['packages']
        
        # Timestamp for this build
        self.build_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🚀 UAP Analysis Suite Builder")
        print(f"Platform: {self.platform}")
        print(f"Version: {APP_VERSION}")
        print(f"Timestamp: {self.build_timestamp}")
        
    def log(self, message, level="INFO"):
        """Enhanced logging with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def check_dependencies(self):
        """Check and install build dependencies"""
        self.log("🔍 Checking build dependencies...")
        
        required_packages = [
            "pyinstaller>=5.0.0",
            "pillow>=9.0.0",
            "setuptools>=60.0.0",
            "wheel>=0.37.0"
        ]
        
        for package in required_packages:
            try:
                # Try to import the package
                import_name = package.split(">=")[0].replace("-", "_")
                if import_name == "pyinstaller":
                    import PyInstaller
                elif import_name == "pillow":
                    import PIL
                else:
                    __import__(import_name)
                self.log(f"✅ {package} available")
            except ImportError:
                self.log(f"📦 Installing {package}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True)
                
    def clean_build_dirs(self):
        """Clean previous build artifacts"""
        if self.clean:
            self.log("🧹 Cleaning previous build artifacts...")
            
            dirs_to_clean = [
                PROJECT_DIR / "build",
                PROJECT_DIR / "dist", 
                self.build_dir,
                self.platform_dir,
                PROJECT_DIR / "__pycache__",
            ]
            
            for dir_path in dirs_to_clean:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    self.log(f"Cleaned: {dir_path}")
                    
            # Clean Python cache files
            for cache_file in PROJECT_DIR.rglob("*.pyc"):
                cache_file.unlink()
            for cache_dir in PROJECT_DIR.rglob("__pycache__"):
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
                    
    def prepare_icons(self):
        """Prepare platform-specific icons"""
        self.log("🎨 Preparing icons...")
        
        icon_script = PROJECT_DIR / "scripts" / "prepare_icons.py"
        if icon_script.exists():
            result = subprocess.run([
                sys.executable, str(icon_script)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ Icons prepared successfully")
            else:
                self.log(f"⚠️  Icon preparation failed: {result.stderr}", "WARN")
        else:
            self.log("⚠️  Icon preparation script not found", "WARN")
            
    def build_gui_application(self):
        """Build the main GUI application"""
        self.log("🏗️  Building GUI application...")
        
        config = get_current_platform_config()
        
        # PyInstaller arguments
        args = [
            "pyinstaller",
            "--clean",
            "--noconfirm",
            f"--distpath={self.platform_dir}",
            f"--workpath={self.build_dir / 'work' / 'gui'}",
            f"--name={config['name']}",
        ]
        
        # Add icon if available
        if config.get('icon') and Path(config['icon']).exists():
            args.extend(["--icon", config['icon']])
            
        # Console vs windowed
        if config.get('console', True):
            args.append("--console")
        else:
            args.append("--windowed")
            
        # Add data files
        for src, dst in config['datas']:
            args.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
            
        # Hidden imports
        for import_name in config['hiddenimports']:
            args.extend(["--hidden-import", import_name])
            
        # Exclude modules
        for exclude in config['excludes']:
            args.extend(["--exclude-module", exclude])
            
        # Platform-specific options
        if self.platform == "darwin":
            if config.get('target_arch'):
                args.extend(["--target-arch", config['target_arch']])
            if config.get('bundle_identifier'):
                args.extend(["--osx-bundle-identifier", config['bundle_identifier']])
                
        # Main script
        args.append(str(MAIN_SCRIPT))
        
        # Run PyInstaller
        self.log(f"Running: {' '.join(args)}")
        result = subprocess.run(args, cwd=PROJECT_DIR)
        
        if result.returncode == 0:
            self.log("✅ GUI application built successfully")
            return True
        else:
            self.log("❌ GUI application build failed", "ERROR")
            return False
            
    def build_console_application(self):
        """Build the command-line application"""
        self.log("⌨️  Building console application...")
        
        config = get_console_config()
        
        # PyInstaller arguments for console app
        args = [
            "pyinstaller",
            "--clean",
            "--noconfirm",
            "--console",
            f"--distpath={self.platform_dir / 'console'}",
            f"--workpath={self.build_dir / 'work' / 'console'}",
            f"--name={config['name']}",
        ]
        
        # Add data files
        for src, dst in config['datas']:
            args.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
            
        # Hidden imports
        for import_name in config['hiddenimports']:
            args.extend(["--hidden-import", import_name])
            
        # Main script
        args.append(str(CONSOLE_SCRIPT))
        
        # Run PyInstaller
        result = subprocess.run(args, cwd=PROJECT_DIR)
        
        if result.returncode == 0:
            self.log("✅ Console application built successfully")
            return True
        else:
            self.log("❌ Console application build failed", "ERROR")
            return False
            
    def create_build_info(self):
        """Create build information file"""
        build_info = {
            'app_name': APP_NAME,
            'version': APP_VERSION,
            'platform': self.platform,
            'build_timestamp': self.build_timestamp,
            'python_version': sys.version,
            'build_machine': platform.node(),
            'architecture': platform.machine(),
        }
        
        build_info_path = self.platform_dir / 'build_info.json'
        with open(build_info_path, 'w') as f:
            json.dump(build_info, f, indent=2)
            
        self.log(f"📋 Build info saved to: {build_info_path}")
        
    def test_builds(self):
        """Test the built applications"""
        self.log("🧪 Testing built applications...")
        
        # Test GUI app
        gui_app = self.find_executable(self.platform_dir)
        if gui_app:
            self.log(f"Found GUI executable: {gui_app}")
            # TODO: Add basic smoke test
            
        # Test console app
        console_app = self.find_executable(self.platform_dir / 'console')
        if console_app:
            self.log(f"Found console executable: {console_app}")
            # TODO: Add version check test
            
    def find_executable(self, search_dir):
        """Find the main executable in a directory"""
        if not search_dir.exists():
            return None
            
        patterns = ["*.exe"] if self.platform == "windows" else ["*"]
        
        for pattern in patterns:
            for executable in search_dir.glob(pattern):
                if executable.is_file() and (
                    executable.suffix.lower() in ['.exe', '.app'] or 
                    os.access(executable, os.X_OK)
                ):
                    return executable
        return None
        
    def package_distribution(self):
        """Create distribution packages"""
        self.log("📦 Creating distribution packages...")
        
        # Create timestamped package directory
        package_name = f"{APP_NAME.replace(' ', '_')}_v{APP_VERSION}_{self.platform}_{self.build_timestamp}"
        package_dir = self.packages_dir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy built applications
        if self.platform_dir.exists():
            # Copy main application
            for item in self.platform_dir.iterdir():
                if item.name not in ['console', 'build_info.json']:
                    if item.is_dir():
                        shutil.copytree(item, package_dir / item.name, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, package_dir)
            
            # Copy console application if it exists
            console_dir = self.platform_dir / 'console'
            if console_dir.exists():
                shutil.copytree(console_dir, package_dir / 'console', dirs_exist_ok=True)
            
        # Copy documentation and assets
        docs_to_copy = ['README.md', 'LICENSE', 'SECURITY.md', 'TECHSTACK.md']
        for doc in docs_to_copy:
            doc_path = PROJECT_DIR / doc
            if doc_path.exists():
                shutil.copy2(doc_path, package_dir)
                
        # Copy configuration files
        configs_dir = PROJECT_DIR / 'configs'
        if configs_dir.exists():
            shutil.copytree(configs_dir, package_dir / 'configs', dirs_exist_ok=True)
            
        # Copy example data
        examples_dir = PROJECT_DIR / 'examples'
        if examples_dir.exists():
            shutil.copytree(examples_dir, package_dir / 'examples', dirs_exist_ok=True)
            
        # Copy build info
        build_info_path = self.platform_dir / 'build_info.json'
        if build_info_path.exists():
            shutil.copy2(build_info_path, package_dir)
            
        self.log(f"📦 Package created: {package_dir}")
        return package_dir
        
    def run_full_build(self):
        """Execute the complete build process"""
        try:
            self.log("🚀 Starting full build process...")
            
            # Step 1: Check dependencies
            self.check_dependencies()
            
            # Step 2: Clean previous builds
            self.clean_build_dirs()
            
            # Step 3: Prepare icons
            self.prepare_icons()
            
            # Step 4: Build applications
            gui_success = self.build_gui_application()
            console_success = self.build_console_application()
            
            if not (gui_success or console_success):
                raise RuntimeError("All builds failed")
                
            # Step 5: Create build info
            self.create_build_info()
            
            # Step 6: Test builds
            self.test_builds()
            
            # Step 7: Package distribution
            package_dir = self.package_distribution()
            
            self.log("🎉 Build process completed successfully!")
            self.log(f"📦 Package location: {package_dir}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Build process failed: {e}", "ERROR")
            return False

def main():
    parser = argparse.ArgumentParser(description="UAP Analysis Suite Builder")
    parser.add_argument("--clean", action="store_true", 
                       help="Clean previous build artifacts")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--gui-only", action="store_true",
                       help="Build only the GUI application")
    parser.add_argument("--console-only", action="store_true",
                       help="Build only the console application")
    
    args = parser.parse_args()
    
    # Create builder
    builder = UAP_Builder(clean=args.clean, verbose=args.verbose)
    
    # Run appropriate build
    if args.gui_only:
        success = builder.build_gui_application()
    elif args.console_only:
        success = builder.build_console_application()
    else:
        success = builder.run_full_build()
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())