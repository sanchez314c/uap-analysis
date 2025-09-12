#!/usr/bin/env python3
"""
Windows-Specific Build and Installer Creation Script
Creates executables, MSI packages, and NSIS installers for Windows
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
import json
import uuid
from datetime import datetime

# Import build configuration
sys.path.append(str(Path(__file__).parent.parent))
from build_config import (
    APP_NAME, APP_VERSION, BUNDLE_ID, COPYRIGHT, AUTHOR,
    BUILD_DIRS, PROJECT_DIR, ASSETS_DIR
)

class WindowsBuilder:
    def __init__(self, create_msi=True, create_nsis=True):
        self.create_msi = create_msi
        self.create_nsis = create_nsis
        
        self.windows_dir = BUILD_DIRS['windows']
        self.packages_dir = BUILD_DIRS['packages']
        
        # Windows-specific paths
        self.wix_path = self.find_wix_installation()
        self.nsis_path = self.find_nsis_installation()
        
        print(f"🪟 Windows Builder for {APP_NAME} v{APP_VERSION}")
        
    def log(self, message, level="INFO"):
        """Enhanced logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def find_wix_installation(self):
        """Find WiX Toolset installation"""
        potential_paths = [
            Path(r"C:\Program Files (x86)\WiX Toolset v3.11\bin"),
            Path(r"C:\Program Files\WiX Toolset v3.11\bin"),
            Path(r"C:\Tools\WiX\bin"),
        ]
        
        for path in potential_paths:
            if (path / "candle.exe").exists() and (path / "light.exe").exists():
                self.log(f"✅ Found WiX at: {path}")
                return path
                
        self.log("⚠️  WiX Toolset not found. MSI creation will be skipped.", "WARN")
        return None
        
    def find_nsis_installation(self):
        """Find NSIS installation"""
        potential_paths = [
            Path(r"C:\Program Files (x86)\NSIS"),
            Path(r"C:\Program Files\NSIS"),
            Path(r"C:\Tools\NSIS"),
        ]
        
        for path in potential_paths:
            if (path / "makensis.exe").exists():
                self.log(f"✅ Found NSIS at: {path}")
                return path
                
        self.log("⚠️  NSIS not found. NSIS installer creation will be skipped.", "WARN")
        return None
        
    def create_version_info_rc(self):
        """Create version info resource file"""
        self.log("📋 Creating version info resource...")
        
        # Convert version to tuple format
        version_parts = APP_VERSION.split('.')
        while len(version_parts) < 4:
            version_parts.append('0')
        version_tuple = ','.join(version_parts)
        
        version_rc_content = f'''#include <windows.h>

VS_VERSION_INFO VERSIONINFO
FILEVERSION {version_tuple}
PRODUCTVERSION {version_tuple}
FILEFLAGSMASK 0x3fL
FILEFLAGS 0x0L
FILEOS 0x40004L
FILETYPE 0x1L
FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904b0"
        BEGIN
            VALUE "CompanyName", "{AUTHOR}"
            VALUE "FileDescription", "Advanced Scientific Analysis Tool for Unidentified Aerial Phenomena"
            VALUE "FileVersion", "{APP_VERSION}"
            VALUE "InternalName", "{APP_NAME.replace(' ', '')}"
            VALUE "LegalCopyright", "{COPYRIGHT}"
            VALUE "OriginalFilename", "{APP_NAME.replace(' ', '_')}.exe"
            VALUE "ProductName", "{APP_NAME}"
            VALUE "ProductVersion", "{APP_VERSION}"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x409, 1200
    END
END
'''
        
        version_rc_path = self.windows_dir / "version.rc"
        version_rc_path.write_text(version_rc_content)
        
        self.log(f"✅ Version resource created: {version_rc_path}")
        return version_rc_path
        
    def create_wix_installer(self, exe_path):
        """Create MSI installer using WiX Toolset"""
        if not self.wix_path or not self.create_msi:
            self.log("⏭️  Skipping MSI creation")
            return None
            
        self.log("📦 Creating MSI installer with WiX...")
        
        # Generate unique GUIDs for this build
        product_guid = str(uuid.uuid4()).upper()
        upgrade_guid = str(uuid.uuid5(uuid.NAMESPACE_DNS, BUNDLE_ID)).upper()
        
        # Create WiX source file
        wxs_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Id="{product_guid}"
           Name="{APP_NAME}"
           Language="1033"
           Version="{APP_VERSION}"
           Manufacturer="{AUTHOR}"
           UpgradeCode="{upgrade_guid}">
    
    <Package Id="*"
             Keywords="Installer"
             Description="{APP_NAME} Installer"
             Comments="{APP_NAME} v{APP_VERSION}"
             Manufacturer="{AUTHOR}"
             InstallScope="perMachine"
             InstallerVersion="200"
             Compressed="yes" />
    
    <MajorUpgrade DowngradeErrorMessage="A newer version of [ProductName] is already installed." />
    
    <MediaTemplate EmbedCab="yes" />
    
    <Feature Id="ProductFeature" Title="{APP_NAME}" Level="1">
      <ComponentGroupRef Id="ProductComponents" />
      <ComponentRef Id="ApplicationShortcut" />
      <ComponentRef Id="DesktopShortcut" />
    </Feature>
    
    <!-- Directory structure -->
    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="ProgramFilesFolder">
        <Directory Id="INSTALLFOLDER" Name="{APP_NAME}" />
      </Directory>
      <Directory Id="ProgramMenuFolder">
        <Directory Id="ApplicationProgramsFolder" Name="{APP_NAME}" />
      </Directory>
      <Directory Id="DesktopFolder" Name="Desktop" />
    </Directory>
    
    <!-- Components -->
    <ComponentGroup Id="ProductComponents" Directory="INSTALLFOLDER">
      <Component Id="MainExecutable" Guid="{str(uuid.uuid4()).upper()}">
        <File Id="MainExe" Source="{exe_path}" KeyPath="yes" />
      </Component>
      
      <!-- Add configuration files -->
      <Component Id="ConfigFiles" Guid="{str(uuid.uuid4()).upper()}">
        <File Id="AnalysisConfig" Source="{PROJECT_DIR / 'configs' / 'analysis_config.yaml'}" />
      </Component>
      
      <!-- Add documentation -->
      <Component Id="Documentation" Guid="{str(uuid.uuid4()).upper()}">
        <File Id="ReadmeFile" Source="{PROJECT_DIR / 'README.md'}" />
        <File Id="LicenseFile" Source="{PROJECT_DIR / 'LICENSE'}" />
      </Component>
    </ComponentGroup>
    
    <!-- Start Menu Shortcut -->
    <Component Id="ApplicationShortcut" Directory="ApplicationProgramsFolder" Guid="{str(uuid.uuid4()).upper()}">
      <Shortcut Id="ApplicationStartMenuShortcut"
                Name="{APP_NAME}"
                Description="{APP_NAME} - UAP Video Analysis"
                Target="[INSTALLFOLDER]{exe_path.name}"
                WorkingDirectory="INSTALLFOLDER" />
      <RemoveFolder Id="ApplicationProgramsFolder" On="uninstall" />
      <RegistryValue Root="HKCU" Key="Software\\{AUTHOR}\\{APP_NAME}" Name="installed" Type="integer" Value="1" KeyPath="yes" />
    </Component>
    
    <!-- Desktop Shortcut -->
    <Component Id="DesktopShortcut" Directory="DesktopFolder" Guid="{str(uuid.uuid4()).upper()}">
      <Shortcut Id="ApplicationDesktopShortcut"
                Name="{APP_NAME}"
                Description="{APP_NAME} - UAP Video Analysis"
                Target="[INSTALLFOLDER]{exe_path.name}"
                WorkingDirectory="INSTALLFOLDER" />
      <RegistryValue Root="HKCU" Key="Software\\{AUTHOR}\\{APP_NAME}" Name="desktop_shortcut" Type="integer" Value="1" KeyPath="yes" />
    </Component>
    
    <!-- UI configuration -->
    <UI>
      <UIRef Id="WixUI_InstallDir" />
      <Publish Dialog="WelcomeDlg"
               Control="Next"
               Event="NewDialog"
               Value="InstallDirDlg"
               Order="2">1</Publish>
      <Publish Dialog="InstallDirDlg"
               Control="Back"
               Event="NewDialog"
               Value="WelcomeDlg"
               Order="2">1</Publish>
    </UI>
    
    <Property Id="WIXUI_INSTALLDIR" Value="INSTALLFOLDER" />
    
  </Product>
</Wix>'''
        
        wxs_path = self.windows_dir / f"{APP_NAME.replace(' ', '_')}.wxs"
        wxs_path.write_text(wxs_content)
        
        # Compile with Candle
        wixobj_path = self.windows_dir / f"{APP_NAME.replace(' ', '_')}.wixobj"
        candle_cmd = [
            str(self.wix_path / "candle.exe"),
            "-out", str(wixobj_path),
            str(wxs_path)
        ]
        
        try:
            subprocess.run(candle_cmd, check=True, cwd=self.windows_dir)
            self.log("✅ WiX compilation successful")
        except subprocess.CalledProcessError as e:
            self.log(f"❌ WiX compilation failed: {e}", "ERROR")
            return None
            
        # Link with Light
        msi_name = f"{APP_NAME.replace(' ', '_')}_v{APP_VERSION}_Windows.msi"
        msi_path = self.packages_dir / msi_name
        
        light_cmd = [
            str(self.wix_path / "light.exe"),
            "-ext", "WixUIExtension",
            "-out", str(msi_path),
            str(wixobj_path)
        ]
        
        try:
            subprocess.run(light_cmd, check=True, cwd=self.windows_dir)
            self.log(f"✅ MSI installer created: {msi_path}")
            
            # Get MSI size
            msi_size = msi_path.stat().st_size / (1024 * 1024)  # MB
            self.log(f"📏 MSI size: {msi_size:.1f} MB")
            
            return msi_path
        except subprocess.CalledProcessError as e:
            self.log(f"❌ MSI linking failed: {e}", "ERROR")
            return None
            
    def create_nsis_installer(self, exe_path):
        """Create NSIS installer"""
        if not self.nsis_path or not self.create_nsis:
            self.log("⏭️  Skipping NSIS installer creation")
            return None
            
        self.log("📦 Creating NSIS installer...")
        
        # Create NSIS script
        nsi_content = f'''!define APP_NAME "{APP_NAME}"
!define APP_VERSION "{APP_VERSION}"
!define APP_PUBLISHER "{AUTHOR}"
!define APP_URL "https://github.com/your-org/UAP-Analysis"
!define APP_DESCRIPTION "Advanced Scientific Analysis Tool for Unidentified Aerial Phenomena"

; Main installer attributes
Name "${{APP_NAME}}"
OutFile "{self.packages_dir / f'{APP_NAME.replace(" ", "_")}_v{APP_VERSION}_Windows_Setup.exe'}"
InstallDir "$PROGRAMFILES\\${{APP_NAME}}"
InstallDirRegKey HKLM "Software\\${{APP_PUBLISHER}}\\${{APP_NAME}}" "InstallDir"
RequestExecutionLevel admin

; Version information
VIProductVersion "{APP_VERSION}.0"
VIAddVersionKey ProductName "${{APP_NAME}}"
VIAddVersionKey ProductVersion "${{APP_VERSION}}"
VIAddVersionKey CompanyName "${{APP_PUBLISHER}}"
VIAddVersionKey FileDescription "${{APP_DESCRIPTION}}"
VIAddVersionKey FileVersion "${{APP_VERSION}}"
VIAddVersionKey LegalCopyright "{COPYRIGHT}"

; Modern UI
!include "MUI2.nsh"

; Interface settings
!define MUI_ABORTWARNING
!define MUI_ICON "${{NSISDIR}}\\Contrib\\Graphics\\Icons\\modern-install.ico"
!define MUI_UNICON "${{NSISDIR}}\\Contrib\\Graphics\\Icons\\modern-uninstall.ico"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "{PROJECT_DIR / 'LICENSE'}"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Language
!insertmacro MUI_LANGUAGE "English"

; Installer sections
Section "Main Application" SecMain
  SectionIn RO
  
  SetOutPath "$INSTDIR"
  
  ; Add main executable
  File "{exe_path}"
  
  ; Add configuration files
  SetOutPath "$INSTDIR\\configs"
  File /r "{PROJECT_DIR / 'configs'}\\*.*"
  
  ; Add documentation
  SetOutPath "$INSTDIR\\docs"
  File "{PROJECT_DIR / 'README.md'}"
  File "{PROJECT_DIR / 'LICENSE'}"
  File "{PROJECT_DIR / 'SECURITY.md'}"
  
  ; Registry entries
  WriteRegStr HKLM "Software\\${{APP_PUBLISHER}}\\${{APP_NAME}}" "InstallDir" "$INSTDIR"
  WriteRegStr HKLM "Software\\${{APP_PUBLISHER}}\\${{APP_NAME}}" "Version" "${{APP_VERSION}}"
  
  ; Add to Add/Remove Programs
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "DisplayName" "${{APP_NAME}}"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "UninstallString" '"$INSTDIR\\Uninstall.exe"'
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "DisplayIcon" "$INSTDIR\\{exe_path.name}"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "DisplayVersion" "${{APP_VERSION}}"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "Publisher" "${{APP_PUBLISHER}}"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "URLInfoAbout" "${{APP_URL}}"
  
  ; Create uninstaller
  WriteUninstaller "$INSTDIR\\Uninstall.exe"
  
SectionEnd

Section "Start Menu Shortcuts" SecStartMenu
  CreateDirectory "$SMPROGRAMS\\${{APP_NAME}}"
  CreateShortCut "$SMPROGRAMS\\${{APP_NAME}}\\${{APP_NAME}}.lnk" "$INSTDIR\\{exe_path.name}"
  CreateShortCut "$SMPROGRAMS\\${{APP_NAME}}\\Uninstall.lnk" "$INSTDIR\\Uninstall.exe"
SectionEnd

Section "Desktop Shortcut" SecDesktop
  CreateShortCut "$DESKTOP\\${{APP_NAME}}.lnk" "$INSTDIR\\{exe_path.name}"
SectionEnd

; Section descriptions
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${{SecMain}} "Core application files (required)"
  !insertmacro MUI_DESCRIPTION_TEXT ${{SecStartMenu}} "Start menu shortcuts"
  !insertmacro MUI_DESCRIPTION_TEXT ${{SecDesktop}} "Desktop shortcut"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; Uninstaller
Section "Uninstall"
  
  ; Remove files
  Delete "$INSTDIR\\{exe_path.name}"
  Delete "$INSTDIR\\Uninstall.exe"
  RMDir /r "$INSTDIR\\configs"
  RMDir /r "$INSTDIR\\docs"
  RMDir "$INSTDIR"
  
  ; Remove shortcuts
  Delete "$SMPROGRAMS\\${{APP_NAME}}\\*.*"
  RMDir "$SMPROGRAMS\\${{APP_NAME}}"
  Delete "$DESKTOP\\${{APP_NAME}}.lnk"
  
  ; Remove registry entries
  DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}"
  DeleteRegKey HKLM "Software\\${{APP_PUBLISHER}}\\${{APP_NAME}}"
  
SectionEnd'''
        
        nsi_path = self.windows_dir / f"{APP_NAME.replace(' ', '_')}.nsi"
        nsi_path.write_text(nsi_content)
        
        # Compile NSIS installer
        makensis_cmd = [
            str(self.nsis_path / "makensis.exe"),
            str(nsi_path)
        ]
        
        try:
            subprocess.run(makensis_cmd, check=True, cwd=self.windows_dir)
            
            installer_name = f"{APP_NAME.replace(' ', '_')}_v{APP_VERSION}_Windows_Setup.exe"
            installer_path = self.packages_dir / installer_name
            
            self.log(f"✅ NSIS installer created: {installer_path}")
            
            # Get installer size
            if installer_path.exists():
                installer_size = installer_path.stat().st_size / (1024 * 1024)  # MB
                self.log(f"📏 NSIS installer size: {installer_size:.1f} MB")
                
            return installer_path
        except subprocess.CalledProcessError as e:
            self.log(f"❌ NSIS compilation failed: {e}", "ERROR")
            return None
            
    def create_portable_package(self, exe_path):
        """Create portable ZIP package"""
        self.log("📦 Creating portable package...")
        
        # Create portable directory
        portable_name = f"{APP_NAME.replace(' ', '_')}_v{APP_VERSION}_Windows_Portable"
        portable_dir = self.packages_dir / portable_name
        
        if portable_dir.exists():
            shutil.rmtree(portable_dir)
        portable_dir.mkdir(parents=True)
        
        # Copy executable
        if exe_path.is_file():
            shutil.copy2(exe_path, portable_dir)
        else:
            # Copy entire directory
            for item in exe_path.iterdir():
                if item.is_dir():
                    shutil.copytree(item, portable_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, portable_dir)
                    
        # Copy configuration files
        configs_src = PROJECT_DIR / 'configs'
        if configs_src.exists():
            shutil.copytree(configs_src, portable_dir / 'configs', dirs_exist_ok=True)
            
        # Copy documentation
        docs_to_copy = ['README.md', 'LICENSE', 'SECURITY.md', 'TECHSTACK.md']
        for doc in docs_to_copy:
            doc_path = PROJECT_DIR / doc
            if doc_path.exists():
                shutil.copy2(doc_path, portable_dir)
                
        # Create batch file launcher
        batch_content = f'''@echo off
cd /d "%~dp0"
"{exe_path.name}" %*
pause
'''
        (portable_dir / f"Run_{APP_NAME.replace(' ', '_')}.bat").write_text(batch_content)
        
        # Create ZIP file
        zip_name = f"{portable_name}.zip"
        zip_path = self.packages_dir / zip_name
        
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', str(portable_dir))
        
        # Remove temporary directory
        shutil.rmtree(portable_dir)
        
        self.log(f"✅ Portable package created: {zip_path}")
        
        # Get ZIP size
        zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
        self.log(f"📏 Portable ZIP size: {zip_size:.1f} MB")
        
        return zip_path
        
    def build_windows_installers(self):
        """Complete Windows build and installer creation process"""
        try:
            # Find the built executable
            gui_build_dir = BUILD_DIRS['windows'] / 'gui'
            
            # Look for executable or build directory
            exe_candidates = list(gui_build_dir.glob("*.exe"))
            if exe_candidates:
                exe_path = exe_candidates[0]
            else:
                # Look for PyInstaller output directory
                built_dirs = [d for d in gui_build_dir.iterdir() if d.is_dir()]
                if built_dirs:
                    exe_path = built_dirs[0]  # Use directory
                    # Look for actual exe inside
                    exe_files = list(exe_path.glob("*.exe"))
                    if exe_files:
                        exe_path = exe_files[0]
                else:
                    self.log("❌ No built executable found. Run main build first.", "ERROR")
                    return False
                    
            self.log(f"💻 Found built executable: {exe_path}")
            
            created_packages = []
            
            # Create version info resource
            self.create_version_info_rc()
            
            # Create MSI installer
            if self.create_msi:
                msi_path = self.create_wix_installer(exe_path)
                if msi_path:
                    created_packages.append(("MSI Installer", msi_path))
                    
            # Create NSIS installer
            if self.create_nsis:
                nsis_path = self.create_nsis_installer(exe_path)
                if nsis_path:
                    created_packages.append(("NSIS Installer", nsis_path))
                    
            # Create portable package
            zip_path = self.create_portable_package(exe_path)
            if zip_path:
                created_packages.append(("Portable ZIP", zip_path))
                
            if created_packages:
                self.log("🎉 Windows installers created successfully!")
                for package_type, path in created_packages:
                    self.log(f"📦 {package_type}: {path}")
                return True
            else:
                self.log("❌ No installers were created", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Windows build failed: {e}", "ERROR")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Windows Builder and Installer Creator")
    parser.add_argument("--no-msi", action="store_true", help="Skip MSI creation")
    parser.add_argument("--no-nsis", action="store_true", help="Skip NSIS installer creation")
    
    args = parser.parse_args()
    
    builder = WindowsBuilder(
        create_msi=not args.no_msi,
        create_nsis=not args.no_nsis
    )
    
    success = builder.build_windows_installers()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())