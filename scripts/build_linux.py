#!/usr/bin/env python3
"""
Linux-Specific Build and Package Creation Script
Creates DEB packages, RPM packages, AppImage, and Snap packages for Linux
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
import json
import tarfile
from datetime import datetime
import re

# Import build configuration
sys.path.append(str(Path(__file__).parent.parent))
from build_config import (
    APP_NAME, APP_VERSION, BUNDLE_ID, COPYRIGHT, AUTHOR,
    BUILD_DIRS, PROJECT_DIR, ASSETS_DIR
)

class LinuxBuilder:
    def __init__(self, create_deb=True, create_rpm=True, create_appimage=True, create_tar=True):
        self.create_deb = create_deb
        self.create_rpm = create_rpm
        self.create_appimage = create_appimage
        self.create_tar = create_tar
        
        self.linux_dir = BUILD_DIRS['linux']
        self.packages_dir = BUILD_DIRS['packages']
        
        # Package metadata
        self.package_name = APP_NAME.lower().replace(' ', '-')
        self.maintainer = f"{AUTHOR} <noreply@uapanalysis.org>"
        self.homepage = "https://github.com/your-org/UAP-Analysis"
        self.description_short = "Advanced UAP Video Analysis Tool"
        self.description_long = """Advanced Scientific Analysis Tool for Unidentified Aerial Phenomena.
 Provides comprehensive video analysis capabilities including motion tracking,
 atmospheric analysis, physics-based validation, and machine learning
 classification for scientific investigation of unexplained aerial phenomena."""
        
        print(f"🐧 Linux Builder for {APP_NAME} v{APP_VERSION}")
        
    def log(self, message, level="INFO"):
        """Enhanced logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def check_build_tools(self):
        """Check availability of Linux packaging tools"""
        self.log("🔍 Checking Linux build tools...")
        
        tools_status = {}
        
        # Check for DEB tools
        if self.create_deb:
            deb_tools = ['dpkg-deb', 'fakeroot']
            tools_status['deb'] = all(
                shutil.which(tool) is not None for tool in deb_tools
            )
            if tools_status['deb']:
                self.log("✅ DEB packaging tools available")
            else:
                self.log("⚠️  DEB tools missing. Install with: sudo apt install dpkg-dev fakeroot", "WARN")
                
        # Check for RPM tools
        if self.create_rpm:
            rpm_tools = ['rpmbuild']
            tools_status['rpm'] = all(
                shutil.which(tool) is not None for tool in rpm_tools
            )
            if tools_status['rpm']:
                self.log("✅ RPM packaging tools available")
            else:
                self.log("⚠️  RPM tools missing. Install with: sudo dnf install rpm-build", "WARN")
                
        # Check for AppImage tools
        if self.create_appimage:
            # We'll download appimagetool if needed
            tools_status['appimage'] = True
            self.log("✅ AppImage packaging available")
            
        return tools_status
        
    def create_desktop_file(self):
        """Create .desktop file for Linux desktop integration"""
        self.log("🖥️  Creating .desktop file...")
        
        desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name={APP_NAME}
Comment={self.description_short}
Exec={self.package_name} %F
Icon={self.package_name}
Terminal=false
StartupNotify=true
Categories=Science;Education;AudioVideo;
MimeType=video/mp4;video/avi;video/quicktime;video/x-msvideo;
Keywords=UAP;video;analysis;science;phenomena;tracking;motion;
StartupWMClass={APP_NAME.replace(' ', '')}
"""
        
        desktop_path = self.linux_dir / f"{self.package_name}.desktop"
        desktop_path.write_text(desktop_content)
        
        self.log(f"✅ Desktop file created: {desktop_path}")
        return desktop_path
        
    def create_deb_package(self, executable_path):
        """Create Debian package"""
        if not self.create_deb:
            self.log("⏭️  Skipping DEB package creation")
            return None
            
        self.log("📦 Creating DEB package...")
        
        # Create package directory structure
        package_version = re.sub(r'[^0-9.]', '', APP_VERSION)  # Clean version for DEB
        deb_name = f"{self.package_name}_{package_version}_amd64"
        deb_dir = self.linux_dir / "deb" / deb_name
        
        if deb_dir.exists():
            shutil.rmtree(deb_dir)
        deb_dir.mkdir(parents=True)
        
        # Create DEBIAN control directory
        debian_dir = deb_dir / "DEBIAN"
        debian_dir.mkdir()
        
        # Create directory structure
        usr_bin = deb_dir / "usr" / "bin"
        usr_share_applications = deb_dir / "usr" / "share" / "applications"
        usr_share_icons = deb_dir / "usr" / "share" / "icons" / "hicolor"
        usr_share_doc = deb_dir / "usr" / "share" / "doc" / self.package_name
        usr_lib = deb_dir / "usr" / "lib" / self.package_name
        
        for path in [usr_bin, usr_share_applications, usr_share_doc, usr_lib]:
            path.mkdir(parents=True)
            
        # Create icon directories
        for size in [16, 22, 24, 32, 48, 64, 128, 256, 512]:
            (usr_share_icons / f"{size}x{size}" / "apps").mkdir(parents=True)
            
        # Install executable
        if executable_path.is_file():
            shutil.copy2(executable_path, usr_bin / self.package_name)
            os.chmod(usr_bin / self.package_name, 0o755)
        else:
            # Copy entire directory for PyInstaller bundle
            shutil.copytree(executable_path, usr_lib, dirs_exist_ok=True)
            
            # Create wrapper script
            wrapper_script = f"""#!/bin/bash
cd /usr/lib/{self.package_name}
exec ./{executable_path.name} "$@"
"""
            wrapper_path = usr_bin / self.package_name
            wrapper_path.write_text(wrapper_script)
            wrapper_path.chmod(0o755)
            
        # Install desktop file
        desktop_file = self.create_desktop_file()
        shutil.copy2(desktop_file, usr_share_applications)
        
        # Install icons
        icons_src = ASSETS_DIR / "icons" / "linux"
        if icons_src.exists():
            for icon_file in icons_src.glob("app_icon_*.png"):
                size_match = re.search(r'(\d+)x(\d+)', icon_file.name)
                if size_match:
                    size = size_match.group(1)
                    dest_dir = usr_share_icons / f"{size}x{size}" / "apps"
                    shutil.copy2(icon_file, dest_dir / f"{self.package_name}.png")
                    
        # Install documentation
        docs_to_copy = ['README.md', 'LICENSE', 'SECURITY.md']
        for doc in docs_to_copy:
            doc_path = PROJECT_DIR / doc
            if doc_path.exists():
                shutil.copy2(doc_path, usr_share_doc)
                
        # Create changelog
        changelog_content = f"""{self.package_name} ({APP_VERSION}) stable; urgency=medium

  * Release version {APP_VERSION}
  * Advanced UAP video analysis capabilities
  * Machine learning and computer vision integration
  * Cross-platform compatibility

 -- {self.maintainer}  {datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')}
"""
        (usr_share_doc / "changelog").write_text(changelog_content)
        
        # Compress changelog
        subprocess.run(['gzip', '-9', str(usr_share_doc / "changelog")], check=True)
        
        # Create copyright file
        copyright_content = f"""Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: {self.package_name}
Upstream-Contact: {self.maintainer}
Source: {self.homepage}

Files: *
Copyright: 2024 {AUTHOR}
License: MIT
 {(PROJECT_DIR / 'LICENSE').read_text().replace(chr(10), chr(10) + ' ')}
"""
        (usr_share_doc / "copyright").write_text(copyright_content)
        
        # Calculate installed size
        installed_size = sum(
            f.stat().st_size for f in deb_dir.rglob('*') if f.is_file()
        ) // 1024  # KB
        
        # Create control file
        control_content = f"""Package: {self.package_name}
Version: {package_version}
Section: science
Priority: optional
Architecture: amd64
Maintainer: {self.maintainer}
Installed-Size: {installed_size}
Depends: libc6 (>= 2.27), libgcc-s1 (>= 3.0), libstdc++6 (>= 5.2)
Homepage: {self.homepage}
Description: {self.description_short}
{self.description_long}
"""
        (debian_dir / "control").write_text(control_content)
        
        # Create postinst script
        postinst_content = f"""#!/bin/bash
set -e

# Update desktop database
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications
fi

# Update icon cache
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q /usr/share/icons/hicolor
fi

exit 0
"""
        postinst_path = debian_dir / "postinst"
        postinst_path.write_text(postinst_content)
        postinst_path.chmod(0o755)
        
        # Create postrm script
        postrm_content = f"""#!/bin/bash
set -e

if [ "$1" = "remove" ] || [ "$1" = "purge" ]; then
    # Update desktop database
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database -q /usr/share/applications
    fi
    
    # Update icon cache
    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        gtk-update-icon-cache -q /usr/share/icons/hicolor
    fi
fi

exit 0
"""
        postrm_path = debian_dir / "postrm"
        postrm_path.write_text(postrm_content)
        postrm_path.chmod(0o755)
        
        # Build DEB package
        deb_file = self.packages_dir / f"{deb_name}.deb"
        
        build_cmd = [
            'fakeroot', 'dpkg-deb', '--build', str(deb_dir), str(deb_file)
        ]
        
        try:
            subprocess.run(build_cmd, check=True)
            self.log(f"✅ DEB package created: {deb_file}")
            
            # Get package size
            deb_size = deb_file.stat().st_size / (1024 * 1024)  # MB
            self.log(f"📏 DEB package size: {deb_size:.1f} MB")
            
            return deb_file
        except subprocess.CalledProcessError as e:
            self.log(f"❌ DEB package creation failed: {e}", "ERROR")
            return None
            
    def create_rpm_package(self, executable_path):
        """Create RPM package"""
        if not self.create_rpm:
            self.log("⏭️  Skipping RPM package creation")
            return None
            
        self.log("📦 Creating RPM package...")
        
        # Setup RPM build environment
        rpm_root = self.linux_dir / "rpm"
        for subdir in ['SOURCES', 'SPECS', 'BUILD', 'RPMS', 'SRPMS']:
            (rpm_root / subdir).mkdir(parents=True, exist_ok=True)
            
        # Create tarball of source
        package_version = re.sub(r'[^0-9.]', '', APP_VERSION)
        source_name = f"{self.package_name}-{package_version}"
        source_dir = rpm_root / "BUILD" / source_name
        
        if source_dir.exists():
            shutil.rmtree(source_dir)
        source_dir.mkdir(parents=True)
        
        # Copy executable and resources
        if executable_path.is_file():
            shutil.copy2(executable_path, source_dir / self.package_name)
        else:
            shutil.copytree(executable_path, source_dir / "lib", dirs_exist_ok=True)
            
        # Copy desktop file
        desktop_file = self.create_desktop_file()
        shutil.copy2(desktop_file, source_dir)
        
        # Copy icons
        icons_dir = source_dir / "icons"
        icons_dir.mkdir()
        icons_src = ASSETS_DIR / "icons" / "linux"
        if icons_src.exists():
            shutil.copytree(icons_src, icons_dir, dirs_exist_ok=True)
            
        # Copy documentation
        for doc in ['README.md', 'LICENSE', 'SECURITY.md']:
            doc_path = PROJECT_DIR / doc
            if doc_path.exists():
                shutil.copy2(doc_path, source_dir)
                
        # Create tarball
        tarball_path = rpm_root / "SOURCES" / f"{source_name}.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(source_dir, arcname=source_name)
            
        # Create RPM spec file
        spec_content = f"""Name:           {self.package_name}
Version:        {package_version}
Release:        1%{{?dist}}
Summary:        {self.description_short}

License:        MIT
URL:            {self.homepage}
Source0:        %{{name}}-%{{version}}.tar.gz

BuildRequires:  gcc
Requires:       glibc

%description
{self.description_long.replace(chr(10), chr(10))}

%prep
%autosetup

%build
# No build required for pre-compiled binary

%install
rm -rf $RPM_BUILD_ROOT

# Install binary
mkdir -p $RPM_BUILD_ROOT%{{_bindir}}
"""
        
        if executable_path.is_file():
            spec_content += f"""install -m 755 {self.package_name} $RPM_BUILD_ROOT%{{_bindir}}/{self.package_name}
"""
        else:
            spec_content += f"""mkdir -p $RPM_BUILD_ROOT%{{_libdir}}/{self.package_name}
cp -r lib/* $RPM_BUILD_ROOT%{{_libdir}}/{self.package_name}/

# Create wrapper script
cat > $RPM_BUILD_ROOT%{{_bindir}}/{self.package_name} << 'EOF'
#!/bin/bash
cd %{{_libdir}}/{self.package_name}
exec ./{executable_path.name} "$@"
EOF
chmod 755 $RPM_BUILD_ROOT%{{_bindir}}/{self.package_name}
"""
        
        spec_content += f"""
# Install desktop file
mkdir -p $RPM_BUILD_ROOT%{{_datadir}}/applications
install -m 644 {self.package_name}.desktop $RPM_BUILD_ROOT%{{_datadir}}/applications/

# Install icons
mkdir -p $RPM_BUILD_ROOT%{{_datadir}}/icons/hicolor
cp -r icons/* $RPM_BUILD_ROOT%{{_datadir}}/icons/hicolor/

# Install documentation
mkdir -p $RPM_BUILD_ROOT%{{_docdir}}/%{{name}}
install -m 644 README.md LICENSE SECURITY.md $RPM_BUILD_ROOT%{{_docdir}}/%{{name}}/

%files
%{{_bindir}}/{self.package_name}
"""
        
        if not executable_path.is_file():
            spec_content += f"%{{_libdir}}/{self.package_name}\n"
            
        spec_content += f"""%{{_datadir}}/applications/{self.package_name}.desktop
%{{_datadir}}/icons/hicolor/*/apps/{self.package_name}.png
%doc %{{_docdir}}/%{{name}}/*

%post
/usr/bin/update-desktop-database &> /dev/null || :
/bin/touch --no-create %{{_datadir}}/icons/hicolor &>/dev/null || :

%postun
/usr/bin/update-desktop-database &> /dev/null || :
if [ $1 -eq 0 ] ; then
    /bin/touch --no-create %{{_datadir}}/icons/hicolor &>/dev/null
    /usr/bin/gtk-update-icon-cache %{{_datadir}}/icons/hicolor &>/dev/null || :
fi

%posttrans
/usr/bin/gtk-update-icon-cache %{{_datadir}}/icons/hicolor &>/dev/null || :

%changelog
* {datetime.now().strftime('%a %b %d %Y')} {AUTHOR} <noreply@uapanalysis.org> - {package_version}-1
- Release version {APP_VERSION}
- Advanced UAP video analysis capabilities
- Machine learning and computer vision integration
"""
        
        spec_path = rpm_root / "SPECS" / f"{self.package_name}.spec"
        spec_path.write_text(spec_content)
        
        # Build RPM
        build_cmd = [
            'rpmbuild',
            '--define', f'_topdir {rpm_root}',
            '-ba', str(spec_path)
        ]
        
        try:
            subprocess.run(build_cmd, check=True)
            
            # Find generated RPM
            rpm_files = list((rpm_root / "RPMS").rglob("*.rpm"))
            if rpm_files:
                rpm_file = rpm_files[0]
                final_rpm = self.packages_dir / rpm_file.name
                shutil.copy2(rpm_file, final_rpm)
                
                self.log(f"✅ RPM package created: {final_rpm}")
                
                # Get package size
                rpm_size = final_rpm.stat().st_size / (1024 * 1024)  # MB
                self.log(f"📏 RPM package size: {rpm_size:.1f} MB")
                
                return final_rpm
            else:
                self.log("❌ No RPM file generated", "ERROR")
                return None
                
        except subprocess.CalledProcessError as e:
            self.log(f"❌ RPM package creation failed: {e}", "ERROR")
            return None
            
    def create_appimage(self, executable_path):
        """Create AppImage universal Linux package"""
        if not self.create_appimage:
            self.log("⏭️  Skipping AppImage creation")
            return None
            
        self.log("📦 Creating AppImage...")
        
        # Download appimagetool if not present
        appimagetool_path = self.linux_dir / "appimagetool"
        if not appimagetool_path.exists():
            self.log("📥 Downloading appimagetool...")
            try:
                import urllib.request
                urllib.request.urlretrieve(
                    "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage",
                    appimagetool_path
                )
                appimagetool_path.chmod(0o755)
                self.log("✅ appimagetool downloaded")
            except Exception as e:
                self.log(f"❌ Failed to download appimagetool: {e}", "ERROR")
                return None
                
        # Create AppDir structure
        appdir_name = f"{APP_NAME.replace(' ', '_')}.AppDir"
        appdir = self.linux_dir / appdir_name
        
        if appdir.exists():
            shutil.rmtree(appdir)
        appdir.mkdir()
        
        # Create directory structure
        usr_bin = appdir / "usr" / "bin"
        usr_lib = appdir / "usr" / "lib"
        usr_share = appdir / "usr" / "share"
        
        for path in [usr_bin, usr_lib, usr_share]:
            path.mkdir(parents=True)
            
        # Install executable
        if executable_path.is_file():
            shutil.copy2(executable_path, usr_bin / self.package_name)
            app_executable = self.package_name
        else:
            # Copy entire PyInstaller bundle
            bundle_dir = usr_lib / self.package_name
            shutil.copytree(executable_path, bundle_dir, dirs_exist_ok=True)
            
            # Create wrapper script
            wrapper_script = f"""#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${{SELF%/*}}
export PATH="${{HERE}}/usr/bin:$PATH"
export LD_LIBRARY_PATH="${{HERE}}/usr/lib:$LD_LIBRARY_PATH"
cd "${{HERE}}/usr/lib/{self.package_name}"
exec ./{executable_path.name} "$@"
"""
            wrapper_path = usr_bin / self.package_name
            wrapper_path.write_text(wrapper_script)
            wrapper_path.chmod(0o755)
            app_executable = self.package_name
            
        # Install desktop file at root
        desktop_file = self.create_desktop_file()
        shutil.copy2(desktop_file, appdir / f"{self.package_name}.desktop")
        
        # Install icon at root (use largest available)
        icon_installed = False
        icons_src = ASSETS_DIR / "icons" / "linux"
        if icons_src.exists():
            # Find largest icon
            icon_files = sorted(
                icons_src.glob("app_icon_*.png"),
                key=lambda x: int(re.search(r'(\d+)x\d+', x.name).group(1)) if re.search(r'(\d+)x\d+', x.name) else 0,
                reverse=True
            )
            if icon_files:
                shutil.copy2(icon_files[0], appdir / f"{self.package_name}.png")
                icon_installed = True
                
        if not icon_installed:
            self.log("⚠️  No icon found for AppImage", "WARN")
            
        # Create AppRun script
        apprun_content = f"""#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${{SELF%/*}}
export PATH="${{HERE}}/usr/bin:$PATH"
export LD_LIBRARY_PATH="${{HERE}}/usr/lib:$LD_LIBRARY_PATH"
export XDG_DATA_DIRS="${{HERE}}/usr/share:$XDG_DATA_DIRS"

cd "$HERE"
exec usr/bin/{app_executable} "$@"
"""
        apprun_path = appdir / "AppRun"
        apprun_path.write_text(apprun_content)
        apprun_path.chmod(0o755)
        
        # Build AppImage
        appimage_name = f"{APP_NAME.replace(' ', '_')}-{APP_VERSION}-x86_64.AppImage"
        appimage_path = self.packages_dir / appimage_name
        
        build_cmd = [
            str(appimagetool_path),
            str(appdir),
            str(appimage_path)
        ]
        
        env = os.environ.copy()
        env['ARCH'] = 'x86_64'
        
        try:
            subprocess.run(build_cmd, check=True, env=env)
            self.log(f"✅ AppImage created: {appimage_path}")
            
            # Get AppImage size
            appimage_size = appimage_path.stat().st_size / (1024 * 1024)  # MB
            self.log(f"📏 AppImage size: {appimage_size:.1f} MB")
            
            return appimage_path
        except subprocess.CalledProcessError as e:
            self.log(f"❌ AppImage creation failed: {e}", "ERROR")
            return None
            
    def create_tar_package(self, executable_path):
        """Create portable tar.gz package"""
        if not self.create_tar:
            self.log("⏭️  Skipping tar package creation")
            return None
            
        self.log("📦 Creating portable tar.gz package...")
        
        # Create package directory
        tar_name = f"{APP_NAME.replace(' ', '_')}_v{APP_VERSION}_Linux_x64"
        tar_dir = self.linux_dir / tar_name
        
        if tar_dir.exists():
            shutil.rmtree(tar_dir)
        tar_dir.mkdir()
        
        # Copy executable
        if executable_path.is_file():
            shutil.copy2(executable_path, tar_dir / self.package_name)
        else:
            # Copy entire directory
            for item in executable_path.iterdir():
                if item.is_dir():
                    shutil.copytree(item, tar_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, tar_dir)
                    
        # Create launcher script
        launcher_content = f"""#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "{self.package_name}" ]; then
    ./{self.package_name} "$@"
elif [ -f "{executable_path.name}" ]; then
    ./{executable_path.name} "$@"
else
    echo "Error: Executable not found"
    exit 1
fi
"""
        launcher_path = tar_dir / f"run_{self.package_name}.sh"
        launcher_path.write_text(launcher_content)
        launcher_path.chmod(0o755)
        
        # Copy documentation
        docs_to_copy = ['README.md', 'LICENSE', 'SECURITY.md', 'TECHSTACK.md']
        for doc in docs_to_copy:
            doc_path = PROJECT_DIR / doc
            if doc_path.exists():
                shutil.copy2(doc_path, tar_dir)
                
        # Copy configuration
        configs_src = PROJECT_DIR / 'configs'
        if configs_src.exists():
            shutil.copytree(configs_src, tar_dir / 'configs', dirs_exist_ok=True)
            
        # Create tar.gz
        tar_file = self.packages_dir / f"{tar_name}.tar.gz"
        
        with tarfile.open(tar_file, "w:gz") as tar:
            tar.add(tar_dir, arcname=tar_name)
            
        # Remove temporary directory
        shutil.rmtree(tar_dir)
        
        self.log(f"✅ Tar package created: {tar_file}")
        
        # Get tar size
        tar_size = tar_file.stat().st_size / (1024 * 1024)  # MB
        self.log(f"📏 Tar package size: {tar_size:.1f} MB")
        
        return tar_file
        
    def build_linux_packages(self):
        """Complete Linux build and package creation process"""
        try:
            # Check build tools
            tools_status = self.check_build_tools()
            
            # Find the built executable
            gui_build_dir = BUILD_DIRS['linux'] / 'gui'
            
            # Look for executable or build directory
            exe_candidates = [f for f in gui_build_dir.rglob('*') if f.is_file() and os.access(f, os.X_OK)]
            if exe_candidates:
                executable_path = exe_candidates[0]
                if len(exe_candidates) > 1:
                    # Look for main executable
                    main_candidates = [f for f in exe_candidates if 'uap' in f.name.lower()]
                    if main_candidates:
                        executable_path = main_candidates[0]
            else:
                # Look for PyInstaller output directory
                built_dirs = [d for d in gui_build_dir.iterdir() if d.is_dir()]
                if built_dirs:
                    executable_path = built_dirs[0]  # Use directory
                else:
                    self.log("❌ No built executable found. Run main build first.", "ERROR")
                    return False
                    
            self.log(f"🐧 Found built executable: {executable_path}")
            
            created_packages = []
            
            # Create DEB package
            if self.create_deb and tools_status.get('deb', False):
                deb_path = self.create_deb_package(executable_path)
                if deb_path:
                    created_packages.append(("DEB Package", deb_path))
                    
            # Create RPM package
            if self.create_rpm and tools_status.get('rpm', False):
                rpm_path = self.create_rpm_package(executable_path)
                if rpm_path:
                    created_packages.append(("RPM Package", rpm_path))
                    
            # Create AppImage
            if self.create_appimage:
                appimage_path = self.create_appimage(executable_path)
                if appimage_path:
                    created_packages.append(("AppImage", appimage_path))
                    
            # Create tar.gz package
            if self.create_tar:
                tar_path = self.create_tar_package(executable_path)
                if tar_path:
                    created_packages.append(("Tar Archive", tar_path))
                    
            if created_packages:
                self.log("🎉 Linux packages created successfully!")
                for package_type, path in created_packages:
                    self.log(f"📦 {package_type}: {path}")
                return True
            else:
                self.log("❌ No packages were created", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Linux build failed: {e}", "ERROR")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Linux Builder and Package Creator")
    parser.add_argument("--no-deb", action="store_true", help="Skip DEB package creation")
    parser.add_argument("--no-rpm", action="store_true", help="Skip RPM package creation")
    parser.add_argument("--no-appimage", action="store_true", help="Skip AppImage creation")
    parser.add_argument("--no-tar", action="store_true", help="Skip tar.gz creation")
    
    args = parser.parse_args()
    
    builder = LinuxBuilder(
        create_deb=not args.no_deb,
        create_rpm=not args.no_rpm,
        create_appimage=not args.no_appimage,
        create_tar=not args.no_tar
    )
    
    success = builder.build_linux_packages()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())