#!/usr/bin/env python3
"""
macOS-Specific Build and Installer Creation Script
Creates .app bundles, DMG installers, and handles code signing
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
import json
import plistlib
from datetime import datetime

# Import build configuration
sys.path.append(str(Path(__file__).parent.parent))
from build_config import (
    APP_NAME, APP_VERSION, BUNDLE_ID, COPYRIGHT,
    BUILD_DIRS, PROJECT_DIR, ASSETS_DIR
)

class MacOSBuilder:
    def __init__(self, sign_code=False, notarize=False):
        self.sign_code = sign_code
        self.notarize = notarize
        
        self.macos_dir = BUILD_DIRS['macos']
        self.packages_dir = BUILD_DIRS['packages']
        
        # Code signing identity (set via environment or prompt)
        self.signing_identity = os.environ.get('CODESIGN_IDENTITY')
        self.notarization_profile = os.environ.get('NOTARIZATION_PROFILE')
        
        print(f"🍎 macOS Builder for {APP_NAME} v{APP_VERSION}")
        
    def log(self, message, level="INFO"):
        """Enhanced logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def check_xcode_tools(self):
        """Check if Xcode command line tools are installed"""
        self.log("🔍 Checking Xcode command line tools...")
        
        tools = ['codesign', 'hdiutil', 'iconutil', 'plutil']
        missing_tools = []
        
        for tool in tools:
            result = subprocess.run(['which', tool], capture_output=True)
            if result.returncode != 0:
                missing_tools.append(tool)
            else:
                self.log(f"✅ {tool} available")
                
        if missing_tools:
            self.log(f"❌ Missing tools: {', '.join(missing_tools)}")
            self.log("Install with: xcode-select --install")
            return False
            
        return True
        
    def create_app_bundle(self, built_app_path):
        """Create a proper macOS .app bundle with metadata"""
        self.log("📦 Creating macOS .app bundle...")
        
        if not built_app_path.exists():
            self.log(f"❌ Built app not found at {built_app_path}", "ERROR")
            return None
            
        # Create bundle directory structure
        bundle_name = f"{APP_NAME}.app"
        bundle_path = self.macos_dir / bundle_name
        
        if bundle_path.exists():
            shutil.rmtree(bundle_path)
            
        bundle_path.mkdir(parents=True)
        
        # Create standard bundle directories
        contents_dir = bundle_path / "Contents"
        macos_dir = contents_dir / "MacOS"
        resources_dir = contents_dir / "Resources"
        
        contents_dir.mkdir()
        macos_dir.mkdir()
        resources_dir.mkdir()
        
        # Copy executable
        app_executable = None
        for item in built_app_path.iterdir():
            if item.is_file() and os.access(item, os.X_OK):
                app_executable = item
                break
                
        if app_executable:
            executable_name = APP_NAME.replace(" ", "_")
            shutil.copy2(app_executable, macos_dir / executable_name)
            # Make sure it's executable
            os.chmod(macos_dir / executable_name, 0o755)
        else:
            self.log("❌ No executable found in built app", "ERROR")
            return None
            
        # Copy resources
        for item in built_app_path.iterdir():
            if item != app_executable:
                if item.is_dir():
                    shutil.copytree(item, resources_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, resources_dir)
                    
        # Copy icon
        icon_path = ASSETS_DIR / "icons" / "macos" / "app_icon.icns"
        if icon_path.exists():
            shutil.copy2(icon_path, resources_dir / "app_icon.icns")
        else:
            self.log("⚠️  App icon not found", "WARN")
            
        # Create Info.plist
        self.create_info_plist(contents_dir, executable_name)
        
        self.log(f"✅ App bundle created: {bundle_path}")
        return bundle_path
        
    def create_info_plist(self, contents_dir, executable_name):
        """Create Info.plist for the app bundle"""
        self.log("📋 Creating Info.plist...")
        
        info_plist = {
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': APP_NAME,
            'CFBundleIdentifier': BUNDLE_ID,
            'CFBundleVersion': APP_VERSION,
            'CFBundleShortVersionString': APP_VERSION,
            'CFBundlePackageType': 'APPL',
            'CFBundleSignature': 'UAPA',
            'CFBundleExecutable': executable_name,
            'CFBundleIconFile': 'app_icon.icns',
            'CFBundleInfoDictionaryVersion': '6.0',
            'NSHumanReadableCopyright': COPYRIGHT,
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.13.0',
            'NSRequiresAquaSystemAppearance': False,
            'LSApplicationCategoryType': 'public.app-category.developer-tools',
            
            # Privacy permissions
            'NSCameraUsageDescription': 'UAP Analysis requires camera access for live video analysis',
            'NSMicrophoneUsageDescription': 'UAP Analysis requires microphone access for acoustic analysis',
            'NSDocumentsFolderUsageDescription': 'UAP Analysis needs access to save analysis results',
            'NSDesktopFolderUsageDescription': 'UAP Analysis needs access to load video files from Desktop',
            'NSDownloadsFolderUsageDescription': 'UAP Analysis needs access to load video files from Downloads',
            'NSRemovableVolumesUsageDescription': 'UAP Analysis needs access to load video files from external drives',
            
            # Document types
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'Video Files',
                    'CFBundleTypeRole': 'Viewer',
                    'LSHandlerRank': 'Alternate',
                    'LSItemContentTypes': [
                        'public.movie',
                        'com.apple.quicktime-movie',
                        'public.avi',
                        'public.mpeg-4',
                    ]
                }
            ],
            
            # URL schemes (for future integration)
            'CFBundleURLTypes': [
                {
                    'CFBundleURLName': 'UAP Analysis Protocol',
                    'CFBundleURLSchemes': ['uapanalysis']
                }
            ]
        }
        
        plist_path = contents_dir / "Info.plist"
        with open(plist_path, 'wb') as f:
            plistlib.dump(info_plist, f)
            
        self.log(f"✅ Info.plist created: {plist_path}")
        
    def code_sign_app(self, app_bundle_path):
        """Code sign the app bundle"""
        if not self.sign_code or not self.signing_identity:
            self.log("⏭️  Skipping code signing (not configured)")
            return True
            
        self.log(f"✍️  Code signing with identity: {self.signing_identity}")
        
        # Sign all binaries and frameworks
        sign_command = [
            'codesign',
            '--force',
            '--deep',
            '--sign', self.signing_identity,
            '--options', 'runtime',  # Hardened runtime
            '--entitlements', self.get_entitlements_path(),
            str(app_bundle_path)
        ]
        
        try:
            result = subprocess.run(sign_command, check=True, capture_output=True, text=True)
            self.log("✅ Code signing successful")
            
            # Verify signature
            verify_command = ['codesign', '--verify', '--verbose', str(app_bundle_path)]
            subprocess.run(verify_command, check=True)
            self.log("✅ Code signature verified")
            
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Code signing failed: {e.stderr}", "ERROR")
            return False
            
    def get_entitlements_path(self):
        """Create or return path to entitlements file"""
        entitlements_path = self.macos_dir / "entitlements.plist"
        
        if not entitlements_path.exists():
            entitlements = {
                'com.apple.security.cs.allow-jit': True,
                'com.apple.security.cs.allow-unsigned-executable-memory': True,
                'com.apple.security.cs.disable-library-validation': True,
                'com.apple.security.device.camera': True,
                'com.apple.security.device.microphone': True,
                'com.apple.security.files.user-selected.read-write': True,
                'com.apple.security.files.downloads.read-write': True,
                'com.apple.security.network.client': True,
                'com.apple.security.network.server': True,
            }
            
            with open(entitlements_path, 'wb') as f:
                plistlib.dump(entitlements, f)
                
        return entitlements_path
        
    def create_dmg_installer(self, app_bundle_path):
        """Create a DMG installer with custom background and layout"""
        self.log("💿 Creating DMG installer...")
        
        dmg_name = f"{APP_NAME.replace(' ', '_')}_v{APP_VERSION}_macOS.dmg"
        dmg_path = self.packages_dir / dmg_name
        
        # Remove existing DMG
        if dmg_path.exists():
            dmg_path.unlink()
            
        # Create temporary directory for DMG contents
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dmg_source = temp_path / "dmg_source"
            dmg_source.mkdir()
            
            # Copy app bundle
            app_name = app_bundle_path.name
            shutil.copytree(app_bundle_path, dmg_source / app_name)
            
            # Create Applications symlink
            applications_link = dmg_source / "Applications"
            applications_link.symlink_to("/Applications")
            
            # Copy documentation
            docs_to_include = ['README.md', 'LICENSE', 'SECURITY.md']
            docs_dir = dmg_source / "Documentation"
            docs_dir.mkdir()
            
            for doc in docs_to_include:
                doc_path = PROJECT_DIR / doc
                if doc_path.exists():
                    shutil.copy2(doc_path, docs_dir)
                    
            # Create custom DMG background and DS_Store if assets exist
            self.customize_dmg_appearance(dmg_source)
            
            # Create DMG
            create_dmg_command = [
                'hdiutil', 'create',
                '-volname', APP_NAME,
                '-srcfolder', str(dmg_source),
                '-ov',
                '-format', 'UDZO',
                '-imagekey', 'zlib-level=9',
                str(dmg_path)
            ]
            
            try:
                subprocess.run(create_dmg_command, check=True)
                self.log(f"✅ DMG created: {dmg_path}")
                
                # Get DMG size
                dmg_size = dmg_path.stat().st_size / (1024 * 1024)  # MB
                self.log(f"📏 DMG size: {dmg_size:.1f} MB")
                
                return dmg_path
            except subprocess.CalledProcessError as e:
                self.log(f"❌ DMG creation failed: {e}", "ERROR")
                return None
                
    def customize_dmg_appearance(self, dmg_source):
        """Customize DMG appearance with background and layout"""
        self.log("🎨 Customizing DMG appearance...")
        
        # Create .background directory
        background_dir = dmg_source / ".background"
        background_dir.mkdir()
        
        # Copy background image if it exists
        background_src = ASSETS_DIR / "dmg_background.png"
        if background_src.exists():
            shutil.copy2(background_src, background_dir)
        else:
            # Create a simple background
            self.create_simple_background(background_dir / "dmg_background.png")
            
        # Create DS_Store for window layout (this would normally be created by manually arranging items)
        # For automated builds, we'll set basic properties
        
    def create_simple_background(self, background_path):
        """Create a simple background image for the DMG"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a 600x400 background
            img = Image.new('RGB', (600, 400), color='#f0f0f0')
            draw = ImageDraw.Draw(img)
            
            # Add some simple text
            try:
                # Try to use system font
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font = ImageFont.load_default()
                
            text = f"{APP_NAME}\nVersion {APP_VERSION}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            draw.text(
                ((600 - text_width) // 2, (400 - text_height) // 2),
                text,
                fill='#333333',
                font=font
            )
            
            img.save(background_path)
            self.log("✅ Created simple background image")
            
        except ImportError:
            self.log("⚠️  PIL not available, skipping background creation", "WARN")
            
    def notarize_dmg(self, dmg_path):
        """Submit DMG for notarization with Apple"""
        if not self.notarize or not self.notarization_profile:
            self.log("⏭️  Skipping notarization (not configured)")
            return True
            
        self.log("📤 Submitting for notarization...")
        
        # Submit for notarization
        submit_command = [
            'xcrun', 'notarytool', 'submit',
            str(dmg_path),
            '--keychain-profile', self.notarization_profile,
            '--wait'
        ]
        
        try:
            result = subprocess.run(submit_command, check=True, capture_output=True, text=True)
            self.log("✅ Notarization successful")
            
            # Staple the notarization ticket
            staple_command = ['xcrun', 'stapler', 'staple', str(dmg_path)]
            subprocess.run(staple_command, check=True)
            self.log("✅ Notarization ticket stapled")
            
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Notarization failed: {e.stderr}", "ERROR")
            return False
            
    def build_macos_installer(self):
        """Complete macOS build and installer creation process"""
        try:
            # Check prerequisites
            if not self.check_xcode_tools():
                return False
                
            # Find the built app
            gui_build_dir = BUILD_DIRS['macos']
            built_apps = list(gui_build_dir.glob("*.app"))
            
            if not built_apps:
                # Look for PyInstaller output directory
                built_dirs = [d for d in gui_build_dir.iterdir() if d.is_dir() and d.name != 'console']
                if built_dirs:
                    built_app_path = built_dirs[0]  # Use first directory found
                else:
                    self.log("❌ No built app found. Run main build first.", "ERROR")
                    return False
            else:
                built_app_path = built_apps[0]
                
            self.log(f"📱 Found built app: {built_app_path}")
            
            # Create proper app bundle
            app_bundle = self.create_app_bundle(built_app_path)
            if not app_bundle:
                return False
                
            # Code sign if configured
            if not self.code_sign_app(app_bundle):
                self.log("⚠️  Continuing without code signing", "WARN")
                
            # Create DMG installer
            dmg_path = self.create_dmg_installer(app_bundle)
            if not dmg_path:
                return False
                
            # Notarize if configured
            if not self.notarize_dmg(dmg_path):
                self.log("⚠️  Continuing without notarization", "WARN")
                
            self.log("🎉 macOS installer created successfully!")
            self.log(f"📦 DMG location: {dmg_path}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ macOS build failed: {e}", "ERROR")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="macOS Builder and Installer Creator")
    parser.add_argument("--sign", action="store_true", help="Code sign the application")
    parser.add_argument("--notarize", action="store_true", help="Notarize the DMG")
    parser.add_argument("--identity", help="Code signing identity")
    parser.add_argument("--profile", help="Notarization keychain profile")
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.identity:
        os.environ['CODESIGN_IDENTITY'] = args.identity
    if args.profile:
        os.environ['NOTARIZATION_PROFILE'] = args.profile
        
    builder = MacOSBuilder(sign_code=args.sign, notarize=args.notarize)
    success = builder.build_macos_installer()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())