#!/bin/bash

# Build script for Linux (Ubuntu) UAP Video Analyzer
# Run this on a Linux system or with cross-compilation setup

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✔${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Get the script directory and go to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

print_header "🐧 BUILDING UAP VIDEO ANALYZER FOR LINUX (UBUNTU)"

# Check if we're on Linux
if [ "$(uname)" != "Linux" ]; then
    print_error "This script is designed to run on Linux"
    print_status "For cross-compilation, use Docker or a Linux VM"
    exit 1
fi

# Check dependencies
print_status "Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    print_status "Install with: sudo apt update && sudo apt install python3 python3-pip python3-tk"
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed"
    exit 1
fi

# Check tkinter
python3 -c "import tkinter" 2>/dev/null || {
    print_error "tkinter is not installed"
    print_status "Install with: sudo apt install python3-tk"
    exit 1
}

# Install PyInstaller if not present
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    print_status "Installing PyInstaller..."
    pip3 install pyinstaller
fi

# Install project dependencies
print_status "Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
fi

if [ -f "requirements-linux.txt" ]; then
    pip3 install -r requirements-linux.txt
fi

# Check for icon
if [ ! -f "resources/icon.png" ]; then
    print_error "Icon not found at resources/icon.png"
    print_status "Ensure the icon file exists before building"
    exit 1
fi

# Clean previous builds
print_status "Cleaning previous builds..."
rm -rf build dist

# Build for Linux
print_status "Building Linux executable..."
pyinstaller "UAP Video Analyzer-Linux.spec" --noconfirm

# Create AppImage directory structure (optional)
print_status "Creating AppImage structure (optional)..."
mkdir -p dist/UAP-Video-Analyzer.AppImage/usr/bin
mkdir -p dist/UAP-Video-Analyzer.AppImage/usr/share/applications
mkdir -p dist/UAP-Video-Analyzer.AppImage/usr/share/icons/hicolor/256x256/apps

# Copy files to AppImage structure
cp -r "dist/uap-video-analyzer"/* "dist/UAP-Video-Analyzer.AppImage/usr/bin/"
cp "resources/icon.png" "dist/UAP-Video-Analyzer.AppImage/usr/share/icons/hicolor/256x256/apps/uap-video-analyzer.png"

# Create .desktop file
cat > "dist/UAP-Video-Analyzer.AppImage/usr/share/applications/uap-video-analyzer.desktop" << EOF
[Desktop Entry]
Type=Application
Name=UAP Video Analyzer
Comment=Analyze UAP video footage with advanced computer vision
Exec=uap-video-analyzer
Icon=uap-video-analyzer
Categories=Science;Graphics;Video;
Terminal=false
StartupNotify=true
EOF

# Create run script for Linux
cat > dist/run-uap-analyzer.sh << 'EOF'
#!/bin/bash

# Run UAP Video Analyzer on Linux
# Ensures proper environment and paths

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if we're in the dist directory
if [ -f "$SCRIPT_DIR/uap-video-analyzer" ]; then
    # Run directly from dist directory
    cd "$SCRIPT_DIR"
    ./uap-video-analyzer "$@"
elif [ -f "$SCRIPT_DIR/usr/bin/uap-video-analyzer" ]; then
    # Run from AppImage structure
    cd "$SCRIPT_DIR/usr/bin"
    ./uap-video-analyzer "$@"
else
    echo "Error: Cannot find uap-video-analyzer executable"
    exit 1
fi
EOF

chmod +x dist/run-uap-analyzer.sh

print_header "✅ BUILD COMPLETED SUCCESSFULLY!"
print_success "Linux executable created at:"
print_status "  • dist/uap-video-analyzer (main executable)"
print_status "  • dist/run-uap-analyzer.sh (convenient runner)"

echo ""
print_status "🚀 To run the application:"
print_status "  • Direct: cd dist && ./uap-video-analyzer"
print_status "  • Or: cd dist && ./run-uap-analyzer.sh"

echo ""
print_status "📦 Installation (optional):"
print_status "  • Copy dist/uap-video-analyzer to /usr/local/bin/"
print_status "  • Copy resources/icon.png to /usr/share/icons/"
print_status "  • Copy .desktop file to /usr/share/applications/"

echo ""
print_status "🐧 Ubuntu dependencies (if needed):"
print_status "  sudo apt update"
print_status "  sudo apt install python3 python3-pip python3-tk"
print_status "  sudo apt install libgl1-mesa-glx libegl1-mesa"

print_success "Ready to run on Ubuntu/Linux!"