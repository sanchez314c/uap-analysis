#!/usr/bin/env python3
"""
UAP Analysis Suite - Master Build Script
=======================================

Comprehensive build orchestration for all platforms and package types.
This script coordinates the complete build-compile-dist process.
"""

import os
import sys
import argparse
import subprocess
import platform
import shutil
from pathlib import Path
from datetime import datetime
import json
import concurrent.futures
import threading

# Import our build modules
from build_config import BUILD_DIRS, APP_NAME, APP_VERSION, PROJECT_DIR
from build import UAP_Builder

class MasterBuilder:
    def __init__(self, platforms=None, clean=False, parallel=True, verbose=False):
        self.platforms = platforms or ['current']
        self.clean = clean
        self.parallel = parallel
        self.verbose = verbose
        
        self.current_platform = platform.system().lower()
        self.build_start_time = datetime.now()
        
        # Threading for parallel builds
        self.build_lock = threading.Lock()
        self.build_results = {}
        
        print(f"🚀 UAP Analysis Suite - Master Builder")
        print(f"Version: {APP_VERSION}")
        print(f"Current Platform: {self.current_platform}")
        print(f"Target Platforms: {', '.join(self.platforms)}")
        print(f"Parallel Builds: {'Enabled' if self.parallel else 'Disabled'}")
        print(f"Clean Build: {'Yes' if self.clean else 'No'}")
        print(f"Started: {self.build_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
    def log(self, message, level="INFO", platform="MASTER"):
        """Thread-safe logging with platform context"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self.build_lock:
            print(f"[{timestamp}] {platform}: {level}: {message}")
            
    def validate_environment(self):
        """Validate build environment and dependencies"""
        self.log("🔍 Validating build environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.log("❌ Python 3.8+ required", "ERROR")
            return False
            
        # Check for required files
        required_files = [
            'build.py',
            'build_config.py',
            'scripts/prepare_icons.py',
            'requirements.txt',
            'build_requirements.txt'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (PROJECT_DIR / file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            self.log(f"❌ Missing required files: {', '.join(missing_files)}", "ERROR")
            return False
            
        # Check for main entry point
        main_scripts = [
            'scripts/uap_gui.py',
            'scripts/run_analysis.py'
        ]
        
        if not any((PROJECT_DIR / script).exists() for script in main_scripts):
            self.log("❌ No main entry point found", "ERROR")
            return False
            
        self.log("✅ Build environment validated")
        return True
        
    def install_build_dependencies(self):
        """Install required build dependencies"""
        self.log("📦 Installing build dependencies...")
        
        try:
            # Install core build requirements
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 
                str(PROJECT_DIR / 'build_requirements.txt')
            ], check=True)
            
            self.log("✅ Build dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed to install dependencies: {e}", "ERROR")
            return False
            
    def prepare_build_environment(self):
        """Prepare the build environment"""
        self.log("🔧 Preparing build environment...")
        
        # Create build directories
        for build_dir in BUILD_DIRS.values():
            build_dir.mkdir(parents=True, exist_ok=True)
            
        # Prepare icons for all platforms
        try:
            subprocess.run([
                sys.executable, 
                str(PROJECT_DIR / 'scripts' / 'prepare_icons.py')
            ], check=True)
            self.log("✅ Icons prepared")
        except subprocess.CalledProcessError:
            self.log("⚠️  Icon preparation failed, continuing...", "WARN")
            
        return True
        
    def build_platform(self, target_platform):
        """Build for a specific platform"""
        platform_name = target_platform.upper()
        self.log(f"🏗️  Starting build for {platform_name}", "INFO", platform_name)
        
        try:
            # Step 1: Core application build
            builder = UAP_Builder(clean=self.clean, verbose=self.verbose)
            if not builder.run_full_build():
                raise RuntimeError("Core build failed")
                
            self.log("✅ Core build completed", "INFO", platform_name)
            
            # Step 2: Platform-specific packaging
            if target_platform == 'linux' or (target_platform == 'current' and self.current_platform == 'linux'):
                subprocess.run([
                    sys.executable,
                    str(PROJECT_DIR / 'scripts' / 'build_linux.py')
                ], check=True)
                
            elif target_platform == 'windows' or (target_platform == 'current' and self.current_platform == 'windows'):
                subprocess.run([
                    sys.executable,
                    str(PROJECT_DIR / 'scripts' / 'build_windows.py')
                ], check=True)
                
            elif target_platform == 'macos' or target_platform == 'darwin' or (target_platform == 'current' and self.current_platform == 'darwin'):
                subprocess.run([
                    sys.executable,
                    str(PROJECT_DIR / 'scripts' / 'build_macos.py')
                ], check=True)
                
            else:
                raise RuntimeError(f"Unsupported platform: {target_platform}")
                
            self.log("✅ Platform-specific packaging completed", "INFO", platform_name)
            
            # Step 3: Validation
            packages_dir = BUILD_DIRS['packages']
            package_files = list(packages_dir.glob('*'))
            
            if not package_files:
                raise RuntimeError("No packages were created")
                
            self.log(f"📦 Created {len(package_files)} packages", "INFO", platform_name)
            
            with self.build_lock:
                self.build_results[target_platform] = {
                    'status': 'success',
                    'packages': [str(f) for f in package_files],
                    'package_count': len(package_files)
                }
                
            return True
            
        except Exception as e:
            self.log(f"❌ Build failed: {e}", "ERROR", platform_name)
            
            with self.build_lock:
                self.build_results[target_platform] = {
                    'status': 'failed',
                    'error': str(e),
                    'packages': [],
                    'package_count': 0
                }
            return False
            
    def build_all_platforms(self):
        """Build for all specified platforms"""
        self.log(f"🚀 Starting builds for {len(self.platforms)} platform(s)")
        
        # Expand 'current' platform
        expanded_platforms = []
        for platform_name in self.platforms:
            if platform_name == 'current':
                expanded_platforms.append(self.current_platform)
            else:
                expanded_platforms.append(platform_name)
                
        # Remove duplicates while preserving order
        self.platforms = list(dict.fromkeys(expanded_platforms))
        
        if self.parallel and len(self.platforms) > 1:
            # Parallel builds
            self.log("⚡ Running parallel builds")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.platforms)) as executor:
                future_to_platform = {
                    executor.submit(self.build_platform, platform): platform 
                    for platform in self.platforms
                }
                
                for future in concurrent.futures.as_completed(future_to_platform):
                    platform_name = future_to_platform[future]
                    try:
                        success = future.result()
                    except Exception as e:
                        self.log(f"❌ Unexpected error in {platform_name}: {e}", "ERROR")
                        
        else:
            # Sequential builds
            self.log("🔄 Running sequential builds")
            
            for platform_name in self.platforms:
                self.build_platform(platform_name)
                
    def generate_build_report(self):
        """Generate comprehensive build report"""
        build_end_time = datetime.now()
        total_duration = build_end_time - self.build_start_time
        
        report = {
            'build_info': {
                'app_name': APP_NAME,
                'version': APP_VERSION,
                'start_time': self.build_start_time.isoformat(),
                'end_time': build_end_time.isoformat(),
                'duration_seconds': total_duration.total_seconds(),
                'build_machine': platform.node(),
                'python_version': sys.version
            },
            'platforms': self.build_results,
            'summary': {
                'total_platforms': len(self.platforms),
                'successful_builds': sum(1 for r in self.build_results.values() if r['status'] == 'success'),
                'failed_builds': sum(1 for r in self.build_results.values() if r['status'] == 'failed'),
                'total_packages': sum(r.get('package_count', 0) for r in self.build_results.values())
            }
        }
        
        # Save report
        report_path = BUILD_DIRS['packages'] / f"build_report_{build_end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Display summary
        print("\n" + "=" * 60)
        print("📋 BUILD SUMMARY")
        print("=" * 60)
        print(f"⏱️  Total Duration: {total_duration}")
        print(f"🎯 Platforms Built: {report['summary']['total_platforms']}")
        print(f"✅ Successful: {report['summary']['successful_builds']}")
        print(f"❌ Failed: {report['summary']['failed_builds']}")
        print(f"📦 Total Packages: {report['summary']['total_packages']}")
        
        print("\n📊 PLATFORM RESULTS:")
        for platform, result in self.build_results.items():
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            platform_name = platform.upper()
            
            if result['status'] == 'success':
                print(f"{status_emoji} {platform_name}: {result['package_count']} packages created")
            else:
                print(f"{status_emoji} {platform_name}: {result.get('error', 'Build failed')}")
                
        if report['summary']['total_packages'] > 0:
            print(f"\n📁 All packages saved to: {BUILD_DIRS['packages']}")
            print(f"📋 Build report saved to: {report_path}")
            
        print("=" * 60)
        
        return report
        
    def run_complete_build(self):
        """Execute the complete build process"""
        try:
            # Step 1: Validate environment
            if not self.validate_environment():
                return False
                
            # Step 2: Install dependencies
            if not self.install_build_dependencies():
                return False
                
            # Step 3: Prepare build environment
            if not self.prepare_build_environment():
                return False
                
            # Step 4: Build all platforms
            self.build_all_platforms()
            
            # Step 5: Generate report
            report = self.generate_build_report()
            
            # Step 6: Determine overall success
            success = report['summary']['failed_builds'] == 0
            
            if success:
                self.log("🎉 All builds completed successfully!")
            else:
                self.log(f"⚠️  {report['summary']['failed_builds']} build(s) failed", "WARN")
                
            return success
            
        except KeyboardInterrupt:
            self.log("🛑 Build interrupted by user", "WARN")
            return False
        except Exception as e:
            self.log(f"❌ Unexpected error: {e}", "ERROR")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="UAP Analysis Suite - Master Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_all.py                          # Build for current platform
  python build_all.py --platforms linux        # Build for Linux only
  python build_all.py --platforms linux,windows,macos  # Build for all platforms
  python build_all.py --clean --parallel       # Clean parallel build
  python build_all.py --no-parallel --verbose  # Sequential build with verbose output
        """
    )
    
    parser.add_argument(
        '--platforms', 
        default='current',
        help='Comma-separated list of platforms to build for (current,linux,windows,macos,all)'
    )
    
    parser.add_argument(
        '--clean', 
        action='store_true',
        help='Clean previous build artifacts before building'
    )
    
    parser.add_argument(
        '--no-parallel', 
        action='store_true',
        help='Disable parallel builds (build platforms sequentially)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Parse platforms
    if args.platforms.lower() == 'all':
        platforms = ['linux', 'windows', 'macos']
    elif args.platforms.lower() == 'current':
        platforms = ['current']
    else:
        platforms = [p.strip() for p in args.platforms.split(',')]
        
    # Validate platforms
    valid_platforms = {'current', 'linux', 'windows', 'macos', 'darwin'}
    invalid_platforms = set(platforms) - valid_platforms
    if invalid_platforms:
        print(f"❌ Invalid platforms: {', '.join(invalid_platforms)}")
        print(f"Valid platforms: {', '.join(sorted(valid_platforms))}")
        return 1
        
    # Create master builder
    builder = MasterBuilder(
        platforms=platforms,
        clean=args.clean,
        parallel=not args.no_parallel,
        verbose=args.verbose
    )
    
    # Run build
    success = builder.run_complete_build()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())