#!/bin/bash

# Complete Multi-Platform Python Build Script for UAP Analysis Suite
# Builds Python/CustomTkinter apps for macOS, Windows, and Linux
# Includes automatic temp cleanup and bloat monitoring

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✔${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}[$(date +'%H:%M:%S')] ℹ${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get Python version
get_python_version() {
    python3 --version 2>/dev/null | cut -d' ' -f2 || python --version 2>/dev/null | cut -d' ' -f2 || echo "Not found"
}

# Function to cleanup system temp directories
cleanup_system_temp() {
    print_status "🧹 Cleaning Python build temp directories..."
    
    # macOS temp cleanup
    if [ "$(uname)" = "Darwin" ]; then
        TEMP_DIR=$(find /private/var/folders -name "Temporary*" -type d 2>/dev/null | head -1)
        if [ -n "$TEMP_DIR" ]; then
            PARENT_DIR=$(dirname "$TEMP_DIR")
            BEFORE_SIZE=$(du -sh "$PARENT_DIR" 2>/dev/null | cut -f1)
            
            # Clean up Python/PyInstaller artifacts (older than 1 day)
            find "$PARENT_DIR" -name "pyinstaller-*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
            find "$PARENT_DIR" -name "pip-*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
            find "$PARENT_DIR" -name "python-build-*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
            find "$PARENT_DIR" -name "setuptools-*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
            
            AFTER_SIZE=$(du -sh "$PARENT_DIR" 2>/dev/null | cut -f1)
            print_success "System temp cleanup: $BEFORE_SIZE → $AFTER_SIZE"
        fi
    fi
    
    # Linux temp cleanup
    if [ "$(uname)" = "Linux" ]; then
        if [ -d "/tmp" ]; then
            BEFORE_SIZE=$(du -sh /tmp 2>/dev/null | cut -f1)
            find /tmp -name "pyinstaller-*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
            find /tmp -name "pip-*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
            find /tmp -name "python-*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true
            AFTER_SIZE=$(du -sh /tmp 2>/dev/null | cut -f1)
            print_success "System temp cleanup: $BEFORE_SIZE → $AFTER_SIZE"
        fi
    fi
}

# Function to set custom temp directory
setup_build_temp() {
    BUILD_TEMP_DIR="$SCRIPT_DIR/build-temp"
    mkdir -p "$BUILD_TEMP_DIR"
    export TMPDIR="$BUILD_TEMP_DIR"
    export TMP="$BUILD_TEMP_DIR"
    export TEMP="$BUILD_TEMP_DIR"
    export PYINSTALLER_WORKDIR="$BUILD_TEMP_DIR/pyinstaller"
    print_info "Using custom temp directory: $BUILD_TEMP_DIR"
}

# Function to perform UAP project bloat check
uap_bloat_check() {
    print_status "🔍 Performing UAP Analysis dependencies analysis..."
    
    # Check virtual environment size
    if [ -d "venv" ]; then
        VENV_SIZE=$(du -sh venv/ 2>/dev/null | cut -f1)
        print_info "Virtual environment size: $VENV_SIZE"
    fi
    
    # Check requirements.txt
    if [ -f "requirements.txt" ]; then
        REQ_COUNT=$(grep -v "^#" requirements.txt | grep -v "^$" | wc -l)
        print_info "Active requirements.txt dependencies: $REQ_COUNT"
        
        # Check for heavy UAP analysis dependencies
        HEAVY_DEPS=(opencv-python torch torchvision numpy scipy matplotlib pillow transformers open3d scikit-image)
        for dep in "${HEAVY_DEPS[@]}"; do
            if grep -qi "$dep" requirements.txt; then
                print_warning "⚠️  Heavy scientific dependency detected: $dep"
            fi
        done
    fi
    
    # Check for UAP-specific source complexity
    if [ -d "src" ]; then
        PYTHON_FILES=$(find src scripts -name "*.py" 2>/dev/null | wc -l)
        TOTAL_LINES=$(find src scripts -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
        print_info "UAP Analysis Python files: $PYTHON_FILES ($TOTAL_LINES lines total)"
        
        # Check for specific UAP modules
        for module in analyzers processors visualizers; do
            if [ -d "src/$module" ]; then
                MODULE_FILES=$(find "src/$module" -name "*.py" | wc -l)
                print_info "  $module: $MODULE_FILES files"
            fi
        done
    fi
}

# Function to cleanup build temp after build
cleanup_build_temp() {
    if [ -n "$BUILD_TEMP_DIR" ] && [ -d "$BUILD_TEMP_DIR" ]; then
        print_status "🧹 Cleaning build temp directory..."
        TEMP_SIZE=$(du -sh "$BUILD_TEMP_DIR" 2>/dev/null | cut -f1 || echo "0")
        rm -rf "$BUILD_TEMP_DIR" 2>/dev/null || true
        print_success "Cleaned build temp: $TEMP_SIZE"
    fi
}

# Function to display help
show_help() {
    echo "UAP Analysis Suite - Complete Multi-Platform Build Script"
    echo ""
    echo "Usage: ./compile-build-dist-python.sh [options]"
    echo ""
    echo "Options:"
    echo "  --no-clean         Skip cleaning build artifacts"
    echo "  --no-temp-clean    Skip system temp cleanup"
    echo "  --no-bloat-check   Skip bloat analysis"
    echo "  --platform PLAT    Build for specific platform (mac, win, linux, all)"
    echo "  --onefile          Create single executable file"
    echo "  --windowed         Create windowed app (no console)"
    echo "  --upx              Use UPX compression"
    echo "  --console          Enable console mode (for debugging)"
    echo "  --help             Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./compile-build-dist-python.sh                    # Full build for current platform"
    echo "  ./compile-build-dist-python.sh --platform mac     # macOS only"
    echo "  ./compile-build-dist-python.sh --onefile          # Single file executable"
    echo "  ./compile-build-dist-python.sh --windowed --upx   # Windowed app with compression"
    echo "  ./compile-build-dist-python.sh --console          # With console for debugging"
}

# Parse command line arguments
NO_CLEAN=false
NO_TEMP_CLEAN=false
NO_BLOAT_CHECK=false
PLATFORM="current"
ONEFILE=false
WINDOWED=true  # UAP GUI should default to windowed
USE_UPX=false
CONSOLE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-clean)
            NO_CLEAN=true
            shift
            ;;
        --no-temp-clean)
            NO_TEMP_CLEAN=true
            shift
            ;;
        --no-bloat-check)
            NO_BLOAT_CHECK=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --onefile)
            ONEFILE=true
            shift
            ;;
        --windowed)
            WINDOWED=true
            shift
            ;;
        --console)
            CONSOLE_MODE=true
            WINDOWED=false
            shift
            ;;
        --upx)
            USE_UPX=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Trap to ensure cleanup on exit
trap cleanup_build_temp EXIT

# Check for required tools
print_status "🛸 Checking UAP Analysis build requirements..."

PYTHON_CMD=""
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    print_error "Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(get_python_version)
print_info "Python version: $PYTHON_VERSION"

# Check for pip
if ! command_exists pip && ! command_exists pip3; then
    print_error "pip is not installed. Please install pip first."
    exit 1
fi

# Check for PyInstaller
if ! $PYTHON_CMD -c "import PyInstaller" 2>/dev/null; then
    print_warning "PyInstaller not found. Installing..."
    pip3 install PyInstaller
fi

# Check for UPX if requested
if [ "$USE_UPX" = true ]; then
    if ! command_exists upx; then
        print_warning "UPX not found. Install for better compression:"
        print_info "  macOS: brew install upx"
        print_info "  Linux: sudo apt install upx-ucl"
        USE_UPX=false
    else
        print_success "UPX compression available"
    fi
fi

print_success "All requirements met"

# Cleanup system temp directories first
if [ "$NO_TEMP_CLEAN" = false ]; then
    cleanup_system_temp
fi

# Setup custom build temp directory
setup_build_temp

# Perform bloat check before build
if [ "$NO_BLOAT_CHECK" = false ]; then
    uap_bloat_check
fi

# Step 1: Clean everything if not skipped
if [ "$NO_CLEAN" = false ]; then
    print_status "🧹 Purging all existing builds..."
    rm -rf dist/
    rm -rf build/
    rm -rf build-temp/
    rm -rf __pycache__/
    rm -rf *.spec
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    print_success "All build artifacts purged"
fi

# Step 2: Setup virtual environment (if not exists)
if [ ! -d "venv" ]; then
    print_status "📦 Creating virtual environment for UAP Analysis..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
    print_success "Virtual environment created"
fi

# Step 3: Activate virtual environment and install dependencies
print_status "📦 Installing/updating UAP Analysis dependencies..."

# Activate virtual environment
if [ "$(uname)" = "Darwin" ] || [ "$(uname)" = "Linux" ]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi

# Install/upgrade pip
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        print_error "Failed to install UAP Analysis dependencies"
        exit 1
    fi
fi

# Install build-specific requirements
if [ -f "build_requirements.txt" ]; then
    pip install -r build_requirements.txt
fi

# Ensure PyInstaller is installed in venv
pip install PyInstaller

print_success "UAP Analysis dependencies ready"

# Step 4: Determine build parameters
print_status "🎯 Configuring UAP Analysis build parameters..."

PYINSTALLER_OPTS=""
MAIN_FILE="scripts/uap_gui.py"  # UAP Analysis main GUI file

# Verify main file exists
if [ ! -f "$MAIN_FILE" ]; then
    print_error "Cannot find main UAP GUI file: $MAIN_FILE"
    exit 1
fi

print_info "Main UAP GUI file: $MAIN_FILE"

# Configure PyInstaller options for UAP Analysis
if [ "$ONEFILE" = true ]; then
    PYINSTALLER_OPTS="$PYINSTALLER_OPTS --onefile"
    print_info "Mode: Single file executable"
else
    PYINSTALLER_OPTS="$PYINSTALLER_OPTS --onedir"
    print_info "Mode: Directory bundle"
fi

if [ "$WINDOWED" = true ] && [ "$CONSOLE_MODE" = false ]; then
    PYINSTALLER_OPTS="$PYINSTALLER_OPTS --windowed --noconsole"
    print_info "GUI: Windowed UAP interface (no console)"
elif [ "$CONSOLE_MODE" = true ]; then
    PYINSTALLER_OPTS="$PYINSTALLER_OPTS --console"
    print_info "GUI: Console mode enabled for debugging"
else
    print_info "GUI: Default console mode"
fi

# Add icon if available - check multiple common locations
ICON_FILE=""
if [ "$(uname)" = "Darwin" ]; then
    # macOS icon search order
    for icon_path in "resources/icon.icns" "assets/icons/macos/app_icon.icns" "build-resources/icon.icns" "icon.icns" "assets/icon.icns"; do
        if [ -f "$icon_path" ]; then
            ICON_FILE="$icon_path"
            break
        fi
    done
elif [ "$(uname)" = "Linux" ]; then
    # Linux icon search order  
    for icon_path in "resources/icon.png" "assets/icons/linux/app_icon_256x256.png" "build-resources/icon.png" "icon.png" "assets/icon.png"; do
        if [ -f "$icon_path" ]; then
            ICON_FILE="$icon_path"
            break
        fi
    done
else
    # Windows icon search order
    for icon_path in "resources/icon.ico" "assets/icons/windows/app_icon.ico" "build-resources/icon.ico" "icon.ico" "assets/icon.ico"; do
        if [ -f "$icon_path" ]; then
            ICON_FILE="$icon_path"
            break
        fi
    done
fi

if [ -n "$ICON_FILE" ]; then
    PYINSTALLER_OPTS="$PYINSTALLER_OPTS --icon=$ICON_FILE"
    print_info "Icon: $ICON_FILE"
else
    print_warning "No UAP Analysis icon found for $(uname)"
fi

# Add UAP Analysis specific data directories
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --add-data src:src"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --add-data configs:configs"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --add-data data:data"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --add-data assets:assets"
print_info "UAP Analysis modules: src, configs, data, assets included"

# Configure UAP Analysis hidden imports
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --collect-all tkinter"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --collect-all cv2"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --collect-all numpy"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --collect-all matplotlib"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --collect-all scipy"

# UAP-specific hidden imports
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --hidden-import=src.analyzers"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --hidden-import=src.processors" 
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --hidden-import=src.visualizers"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --hidden-import=src.utils"

# Add UPX compression if available
if [ "$USE_UPX" = true ]; then
    PYINSTALLER_OPTS="$PYINSTALLER_OPTS --upx-dir=$(which upx | xargs dirname)"
    print_info "Compression: UPX enabled"
fi

# Set build directory
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --workpath=$BUILD_TEMP_DIR/pyinstaller"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --distpath=dist"
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --specpath=."

# Set application name
PYINSTALLER_OPTS="$PYINSTALLER_OPTS --name=UAP_Video_Analyzer"

# Step 5: Build the UAP Analysis application
print_status "🏗️ Building UAP Analysis application..."
print_status "Platform: $(uname) ($(uname -m))"
print_status "PyInstaller command: pyinstaller $PYINSTALLER_OPTS $MAIN_FILE"

pyinstaller $PYINSTALLER_OPTS "$MAIN_FILE"
BUILD_RESULT=$?

if [ $BUILD_RESULT -ne 0 ]; then
    print_error "UAP Analysis build failed"
    exit 1
fi

print_success "UAP Analysis build completed successfully"

# Step 6: Platform-specific post-processing
print_status "🔧 Post-processing UAP Analysis for $(uname)..."

if [ "$(uname)" = "Darwin" ]; then
    # macOS specific processing
    print_status "🍎 macOS UAP Analysis app bundle processing..."
    
    if [ -d "dist/UAP_Video_Analyzer.app" ]; then
        print_success "Created macOS UAP Analysis app: dist/UAP_Video_Analyzer.app"
        
        # Make executable
        chmod +x "dist/UAP_Video_Analyzer.app/Contents/MacOS/UAP_Video_Analyzer"
        
        # Create convenience launch script
        cat > "dist/launch_uap_analyzer_macos.sh" << 'EOF'
#!/bin/bash
# Launch UAP Video Analyzer on macOS
cd "$(dirname "$0")"
open UAP_Video_Analyzer.app
EOF
        chmod +x "dist/launch_uap_analyzer_macos.sh"
        print_info "Created macOS launcher: dist/launch_uap_analyzer_macos.sh"
    fi
    
elif [ "$(uname)" = "Linux" ]; then
    # Linux specific processing
    print_status "🐧 Linux UAP Analysis executable processing..."
    
    if [ -d "dist/UAP_Video_Analyzer" ]; then
        # Make executable
        chmod +x "dist/UAP_Video_Analyzer/UAP_Video_Analyzer"
        print_success "Created Linux UAP Analysis executable: dist/UAP_Video_Analyzer/UAP_Video_Analyzer"
        
        # Create desktop file for UAP Analysis
        cat > "dist/UAP_Video_Analyzer.desktop" << EOF
[Desktop Entry]
Name=UAP Video Analyzer
Comment=Advanced Scientific Analysis Tool for Unidentified Aerial Phenomena
Exec=$(pwd)/dist/UAP_Video_Analyzer/UAP_Video_Analyzer
Icon=$(pwd)/assets/icons/linux/app_icon_256x256.png
Type=Application
Categories=Science;Education;AudioVideo;Graphics;
StartupNotify=true
EOF
        print_info "Created desktop file: dist/UAP_Video_Analyzer.desktop"
        
        # Create launch script
        cat > "dist/launch_uap_analyzer_linux.sh" << 'EOF'
#!/bin/bash
# Launch UAP Video Analyzer on Linux
cd "$(dirname "$0")"
./UAP_Video_Analyzer/UAP_Video_Analyzer
EOF
        chmod +x "dist/launch_uap_analyzer_linux.sh"
        print_info "Created Linux launcher: dist/launch_uap_analyzer_linux.sh"
    fi
fi

# Post-build UAP Analysis bloat analysis
if [ "$NO_BLOAT_CHECK" = false ]; then
    print_status "🔍 Post-build UAP Analysis size analysis..."
    
    if [ -d "dist" ]; then
        TOTAL_SIZE=$(du -sh dist/ 2>/dev/null | cut -f1)
        print_info "Total UAP Analysis build output size: $TOTAL_SIZE"
        
        # Find and report on executables
        find dist -type f \( -name "UAP_Video_Analyzer*" -o -name "*.exe" -o -name "*.app" -o -perm +111 \) | while read -r file; do
            if [ -f "$file" ]; then
                SIZE=$(ls -lah "$file" | awk '{print $5}' 2>/dev/null || echo "Unknown")
                NAME=$(basename "$file")
                print_info "  $NAME: $SIZE"
                
                # Warning for large files
                SIZE_MB=$(ls -l "$file" 2>/dev/null | awk '{print int($5/1024/1024)}' || echo "0")
                if [ "$SIZE_MB" -gt 200 ]; then
                    print_warning "⚠️  Large UAP Analysis executable: $NAME ($SIZE)"
                fi
            fi
        done
        
        # Check for UAP-specific scientific library bloat
        if find dist -name "*.so" -o -name "*.dll" -o -name "*.dylib" | head -10 | grep -q "cv2\|numpy\|torch\|scipy"; then
            print_info "✓ Scientific computing libraries detected in UAP Analysis build"
        fi
    fi
fi

# Step 7: Display UAP Analysis build results
print_status "📋 UAP Analysis Build Results Summary:"
echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}  🛸 UAP VIDEO ANALYZER BUILD COMPLETE  🛸${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"

if [ -d "dist" ]; then
    print_success "🎉 UAP Analysis build completed successfully!"
    echo ""
    
    # Display platform-specific results
    if [ "$(uname)" = "Darwin" ]; then
        print_info "🍎 macOS UAP Analysis Build:"
        if [ -d "dist/UAP_Video_Analyzer.app" ]; then
            SIZE=$(du -sh "dist/UAP_Video_Analyzer.app" 2>/dev/null | cut -f1)
            echo "   ✓ App Bundle: dist/UAP_Video_Analyzer.app ($SIZE)"
        fi
        if [ -f "dist/launch_uap_analyzer_macos.sh" ]; then
            echo "   ✓ Launcher: dist/launch_uap_analyzer_macos.sh"
        fi
        
    elif [ "$(uname)" = "Linux" ]; then
        print_info "🐧 Linux UAP Analysis Build:"
        if [ -d "dist/UAP_Video_Analyzer" ]; then
            SIZE=$(du -sh "dist/UAP_Video_Analyzer" 2>/dev/null | cut -f1)
            echo "   ✓ Directory: dist/UAP_Video_Analyzer/ ($SIZE)"
        fi
        if [ -f "dist/UAP_Video_Analyzer.desktop" ]; then
            echo "   ✓ Desktop file: dist/UAP_Video_Analyzer.desktop"
        fi
        if [ -f "dist/launch_uap_analyzer_linux.sh" ]; then
            echo "   ✓ Launcher: dist/launch_uap_analyzer_linux.sh"
        fi
        
    else
        print_info "🪟 Windows UAP Analysis Build:"
        if [ -f "dist/UAP_Video_Analyzer.exe" ]; then
            SIZE=$(ls -lh "dist/UAP_Video_Analyzer.exe" 2>/dev/null | awk '{print $5}')
            echo "   ✓ Executable: dist/UAP_Video_Analyzer.exe ($SIZE)"
        fi
    fi
    
    # Show spec file
    if [ -f "UAP_Video_Analyzer.spec" ]; then
        echo "   ✓ PyInstaller spec: UAP_Video_Analyzer.spec"
    fi
    
else
    print_warning "No dist directory found. UAP Analysis build may have failed."
fi

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
print_success "🛸 UAP Analysis build process finished!"
print_status "📁 All UAP Analysis binaries are in: ./dist/"

# Cleanup recommendations for UAP Analysis
echo ""
print_info "🧹 UAP Analysis Cleanup & Optimization Tips:"
print_info "  • Scientific computing libraries are large but necessary"
print_info "  • Consider --upx compression for smaller binaries"
print_info "  • Use --console mode for debugging UAP analysis issues"
print_info "  • Regular temp cleanup recommended for ML dependencies"

print_status ""
print_info "To run UAP Video Analyzer:"
if [ "$(uname)" = "Darwin" ]; then
    print_info "  macOS: ./dist/launch_uap_analyzer_macos.sh"
    print_info "         or: open dist/UAP_Video_Analyzer.app"
elif [ "$(uname)" = "Linux" ]; then
    print_info "  Linux: ./dist/launch_uap_analyzer_linux.sh"
    print_info "         or: ./dist/UAP_Video_Analyzer/UAP_Video_Analyzer"
else
    print_info "  Windows: ./dist/UAP_Video_Analyzer.exe"
fi

echo ""
print_success "🛸 Ready to analyze UAP footage! 🛸"