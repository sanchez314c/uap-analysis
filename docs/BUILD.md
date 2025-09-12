# UAP Analysis Suite - Build System Documentation

This document provides comprehensive instructions for building the UAP Analysis Suite across all supported platforms.

## 🎯 Overview

The build system creates standalone executables and professional installers for:
- **macOS**: DMG installers with code signing support
- **Windows**: NSIS/MSI installers and portable packages  
- **Linux**: DEB/RPM packages, AppImage, and portable archives

## 📋 Prerequisites

### Common Requirements
- **Python 3.8+** with pip
- **16GB+ RAM** (32GB+ recommended for large builds)
- **10GB+ free disk space** for build artifacts

### Platform-Specific Requirements

#### macOS
- **Xcode Command Line Tools**: `xcode-select --install`
- **Homebrew** (recommended): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- **create-dmg**: `brew install create-dmg`

#### Windows
- **Visual Studio Build Tools** or **Visual Studio Community**
- **WiX Toolset v3.11+** (for MSI installers): Download from [wixtoolset.org](https://wixtoolset.org)
- **NSIS 3.0+** (for NSIS installers): Download from [nsis.sourceforge.io](https://nsis.sourceforge.io)

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y \
  python3-dev \
  python3-pip \
  dpkg-dev \
  fakeroot \
  rpm \
  libc6-dev \
  libgl1-mesa-dev \
  libegl1-mesa-dev \
  libglib2.0-dev \
  libxkbcommon-x11-0 \
  libxcb-icccm4 \
  libxcb-image0 \
  libxcb-keysyms1 \
  libxcb-randr0 \
  libxcb-render-util0 \
  libxcb-xinerama0 \
  libxcb-shape0
```

#### Linux (RedHat/Fedora/CentOS)
```bash
sudo dnf install -y \
  python3-devel \
  python3-pip \
  rpm-build \
  gcc \
  gcc-c++ \
  mesa-libGL-devel \
  mesa-libEGL-devel \
  glib2-devel
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt
pip install -r build_requirements.txt

# Install platform-specific requirements
# Linux:
pip install -r requirements-linux.txt
# macOS:
pip install -r requirements-macos.txt
# Windows: (no additional requirements)
```

### 2. Prepare Icons
Place your icon file (preferably `.icns` for macOS) in the `assets/` directory, then run:
```bash
python scripts/prepare_icons.py
```

### 3. Build for Current Platform
```bash
# Simple build
python build_all.py

# Clean build with verbose output
python build_all.py --clean --verbose
```

### 4. Build for All Platforms
```bash
# Build for all platforms (requires cross-platform setup)
python build_all.py --platforms all

# Build specific platforms
python build_all.py --platforms linux,windows
```

## 🏗️ Build Scripts Overview

### Core Build Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `build_all.py` | Master build orchestrator | `python build_all.py [options]` |
| `build.py` | Core application builder | `python build.py [options]` |
| `build_config.py` | Cross-platform configuration | (imported by other scripts) |

### Platform-Specific Scripts

| Script | Platform | Output Formats |
|--------|----------|----------------|
| `scripts/build_macos.py` | macOS | `.app` bundle, `.dmg` installer |
| `scripts/build_windows.py` | Windows | `.exe` (NSIS), `.msi`, `.zip` (portable) |
| `scripts/build_linux.py` | Linux | `.deb`, `.rpm`, `.AppImage`, `.tar.gz` |
| `scripts/prepare_icons.py` | All | Platform-specific icon formats |

## 📦 Output Formats

### macOS Packages
- **`.dmg`**: Professional disk image installer with custom background
- **`.app`**: Application bundle (generated during DMG creation)

### Windows Packages
- **`.exe`**: NSIS installer with start menu and desktop shortcuts
- **`.msi`**: MSI installer for enterprise deployment (requires WiX)
- **`_Portable.zip`**: Portable package requiring no installation

### Linux Packages
- **`.deb`**: Debian/Ubuntu package with desktop integration
- **`.rpm`**: RedHat/Fedora/CentOS package with desktop integration
- **`.AppImage`**: Universal Linux binary requiring no installation
- **`.tar.gz`**: Portable archive with launcher script

## 🔧 Build Options

### Master Builder (`build_all.py`)
```bash
# Build for current platform only
python build_all.py

# Build for specific platforms
python build_all.py --platforms linux,windows,macos

# Clean build (removes previous artifacts)
python build_all.py --clean

# Sequential build (disable parallelization)
python build_all.py --no-parallel

# Verbose output
python build_all.py --verbose

# Combine options
python build_all.py --clean --platforms all --verbose
```

### Core Builder (`build.py`)
```bash
# Full build (GUI + console applications)
python build.py

# GUI application only
python build.py --gui-only

# Console application only
python build.py --console-only

# Clean build
python build.py --clean

# Verbose output
python build.py --verbose
```

### Platform-Specific Builders

#### macOS Builder
```bash
# Basic build
python scripts/build_macos.py

# With code signing (requires developer certificate)
python scripts/build_macos.py --sign --identity "Developer ID Application: Your Name"

# With notarization (requires Apple ID and app-specific password)
python scripts/build_macos.py --sign --notarize --identity "Your Identity" --profile "YourProfile"
```

#### Windows Builder
```bash
# Build all installer types
python scripts/build_windows.py

# Skip MSI creation (if WiX not available)
python scripts/build_windows.py --no-msi

# Skip NSIS installer (if NSIS not available)
python scripts/build_windows.py --no-nsis
```

#### Linux Builder
```bash
# Build all package types
python scripts/build_linux.py

# Skip specific package types
python scripts/build_linux.py --no-deb --no-rpm
python scripts/build_linux.py --no-appimage --no-tar
```

## 🔒 Code Signing and Notarization

### macOS Code Signing
1. **Obtain Developer Certificate**: Enroll in Apple Developer Program
2. **Install Certificate**: Download and install in Keychain
3. **Set Environment Variables**:
   ```bash
   export CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAM_ID)"
   ```
4. **Build with Signing**:
   ```bash
   python scripts/build_macos.py --sign
   ```

### macOS Notarization
1. **Create App-Specific Password**: Generate at appleid.apple.com
2. **Store in Keychain**:
   ```bash
   xcrun notarytool store-credentials "YourProfile" \
     --apple-id "your@email.com" \
     --team-id "TEAM_ID" \
     --password "app-specific-password"
   ```
3. **Set Environment Variable**:
   ```bash
   export NOTARIZATION_PROFILE="YourProfile"
   ```
4. **Build with Notarization**:
   ```bash
   python scripts/build_macos.py --sign --notarize
   ```

### Windows Code Signing
Windows code signing requires a valid code signing certificate from a trusted CA:
1. **Obtain Certificate**: Purchase from DigiCert, Sectigo, etc.
2. **Install Certificate**: Import into Windows Certificate Store
3. **Modify build script**: Add signing commands to `build_windows.py`

## 🤖 Automated Builds (CI/CD)

### GitHub Actions Workflows

#### Release Builds (`.github/workflows/build-release.yml`)
- **Triggers**: Git tags, manual dispatch
- **Platforms**: All supported platforms
- **Outputs**: GitHub Releases with all installer formats
- **Features**: Automatic versioning, checksums, release notes

#### Nightly Builds (`.github/workflows/nightly-build.yml`)
- **Triggers**: Daily at 2 AM UTC, manual dispatch
- **Platforms**: All supported platforms
- **Outputs**: Pre-release with nightly tag
- **Features**: Automatic cleanup of previous nightly releases

### Manual Release Creation
```bash
# Tag and push for release build
git tag v2.0.0
git push origin v2.0.0

# Manual workflow dispatch via GitHub UI
# Go to Actions → Build and Release → Run workflow
```

## 📁 Build Directory Structure

```
build-compile-dist/
├── macos/                  # macOS build artifacts
│   ├── gui/               # GUI application build
│   ├── console/           # Console application build
│   └── *.dmg              # DMG installer
├── windows/               # Windows build artifacts
│   ├── gui/               # GUI application build
│   ├── console/           # Console application build
│   └── *.exe, *.msi       # Windows installers
├── linux/                 # Linux build artifacts
│   ├── gui/               # GUI application build
│   ├── console/           # Console application build
│   └── deb/, rpm/, etc.   # Package build directories
├── dist/                  # Cross-platform outputs
├── packages/              # Final installer packages
└── build_report_*.json    # Build reports
```

## 🐛 Troubleshooting

### Common Issues

#### Build Fails with "PyInstaller not found"
```bash
pip install pyinstaller>=5.0.0
```

#### macOS: "xcode-select: error: tool 'codesign' requires Xcode"
```bash
xcode-select --install
```

#### Windows: "WiX Toolset not found"
- Download and install WiX Toolset from [wixtoolset.org](https://wixtoolset.org)
- Ensure `candle.exe` and `light.exe` are in PATH

#### Linux: "dpkg-deb: command not found"
```bash
sudo apt-get install dpkg-dev fakeroot
```

#### Large Build Sizes
- Build sizes are expected to be large (100-500MB) due to:
  - Python runtime bundling
  - Scientific computing libraries (NumPy, SciPy, OpenCV)
  - GPU acceleration libraries
  - Machine learning models

#### GPU Acceleration Not Working
- **CUDA**: Ensure NVIDIA drivers and CUDA toolkit are installed
- **Metal MPS**: Requires macOS 12.3+ and Apple Silicon or AMD GPUs
- **OpenCL**: Install appropriate GPU drivers

### Debug Mode
Enable verbose output for debugging:
```bash
python build_all.py --verbose
```

Check build logs in:
- Console output during build
- `build-compile-dist/packages/build_report_*.json`

## 📈 Performance Optimization

### Build Speed
- **Parallel Builds**: Use `--parallel` (default) for multiple platforms
- **Incremental Builds**: Avoid `--clean` for faster subsequent builds
- **SSD Storage**: Use SSD for build directory to improve I/O performance

### Package Size Optimization
- Remove unnecessary dependencies from `requirements.txt`
- Use `--exclude-module` in PyInstaller spec for unused modules
- Compress final packages (already done automatically)

### Memory Usage
- **16GB RAM**: Minimum for basic builds
- **32GB RAM**: Recommended for parallel builds
- **64GB RAM**: Optimal for large video processing and ML models

## 🤝 Contributing

### Adding New Package Formats
1. Extend the appropriate platform builder script
2. Add format-specific configuration to `build_config.py`
3. Update this documentation
4. Add CI/CD workflow integration

### Cross-Platform Compatibility
- Test on actual target platforms when possible
- Use virtual machines for cross-platform testing
- Verify installer functionality, not just creation

### Code Signing Integration
- Add signing support for new platforms
- Document certificate requirements
- Implement automatic signing in CI/CD

## 📚 Additional Resources

- **PyInstaller Documentation**: [pyinstaller.readthedocs.io](https://pyinstaller.readthedocs.io)
- **WiX Toolset Documentation**: [wixtoolset.org/documentation](https://wixtoolset.org/documentation)
- **Apple Code Signing**: [developer.apple.com/documentation/security/notarizing_macos_software_before_distribution](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- **Linux Packaging**: [packaging.python.org](https://packaging.python.org)

## 🎉 Success!

Once your build completes successfully, you'll find all installer packages in the `build-compile-dist/packages/` directory. The build system automatically:

- ✅ Creates platform-appropriate installers
- ✅ Generates checksums for verification
- ✅ Includes documentation and configuration files
- ✅ Provides detailed build reports
- ✅ Optimizes package sizes
- ✅ Handles desktop integration
- ✅ Signs and notarizes when configured

Your UAP Analysis Suite is ready for distribution! 🛸