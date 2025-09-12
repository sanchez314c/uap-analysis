#!/bin/bash

# 🔍 UAP ANALYSIS PYTHON BLOAT CHECK SCRIPT
# Comprehensive analysis of UAP Video Analyzer size and optimization opportunities

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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

print_header() {
    echo ""
    echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════${NC}"
    echo ""
}

# Function to convert bytes to human readable
human_readable() {
    local bytes=$1
    if [ $bytes -gt 1073741824 ]; then
        echo "$(($bytes / 1073741824))GB"
    elif [ $bytes -gt 1048576 ]; then
        echo "$(($bytes / 1048576))MB"
    elif [ $bytes -gt 1024 ]; then
        echo "$(($bytes / 1024))KB"
    else
        echo "${bytes}B"
    fi
}

print_header "🛸 UAP VIDEO ANALYZER BLOAT CHECK"

# Check if in UAP Analysis project
if [ ! -f "requirements.txt" ] && [ ! -d "src" ] && [ ! -d "scripts" ]; then
    print_error "No UAP Analysis project found. Run this in your project root directory."
    exit 1
fi

# 1. Virtual environment analysis
print_header "🐍 VIRTUAL ENVIRONMENT ANALYSIS"

if [ -d "venv" ]; then
    VENV_SIZE=$(du -sb venv 2>/dev/null | cut -f1)
    VENV_SIZE_HR=$(human_readable $VENV_SIZE)
    print_info "Virtual environment size: $VENV_SIZE_HR"
    
    # Size categories for scientific computing
    if [ $VENV_SIZE -gt 5368709120 ]; then  # 5GB
        print_warning "⚠️  VERY LARGE: Virtual env > 5GB - heavy ML libraries detected"
    elif [ $VENV_SIZE -gt 2147483648 ]; then  # 2GB
        print_warning "⚠️  LARGE: Virtual env > 2GB - expected for scientific computing"
    elif [ $VENV_SIZE -gt 1073741824 ]; then  # 1GB
        print_info "📊 MEDIUM: Virtual env > 1GB - normal for UAP analysis"
    else
        print_success "✓ Virtual environment size acceptable"
    fi
    
    echo ""
    print_info "Largest packages in UAP Analysis environment:"
    if [ -d "venv/lib" ]; then
        find venv/lib -name "site-packages" | head -1 | xargs -I {} find {} -maxdepth 1 -type d | while read dir; do
            if [ -d "$dir" ]; then
                SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
                NAME=$(basename "$dir")
                if [[ $NAME != "site-packages" ]] && [[ $NAME != "__pycache__" ]] && [[ $NAME != "*.dist-info" ]]; then
                    echo "  $SIZE - $NAME"
                fi
            fi
        done | sort -hr | head -8
    fi
else
    print_warning "No virtual environment found"
    print_info "Recommendation: Create virtual environment for UAP Analysis dependencies"
fi

# 2. UAP Analysis Dependencies analysis
print_header "📋 UAP ANALYSIS DEPENDENCIES"

if [ -f "requirements.txt" ]; then
    REQ_COUNT=$(grep -v "^#" requirements.txt | grep -v "^$" | wc -l)
    TOTAL_LINES=$(wc -l < requirements.txt)
    print_info "Active dependencies: $REQ_COUNT (total lines: $TOTAL_LINES)"
    
    # Check for UAP-specific heavy dependencies
    UAP_HEAVY_DEPS=(opencv-python torch torchvision tensorflow numpy scipy matplotlib pillow scikit-image open3d transformers)
    
    echo ""
    print_info "Scientific computing dependencies detected:"
    HEAVY_FOUND=0
    for dep in "${UAP_HEAVY_DEPS[@]}"; do
        if grep -qi "^$dep\|^${dep}=\|^${dep}>" requirements.txt; then
            SIZE_ESTIMATE="Unknown"
            case $dep in
                tensorflow*) SIZE_ESTIMATE="~500MB" ;;
                torch*) SIZE_ESTIMATE="~300MB" ;;
                opencv-python*) SIZE_ESTIMATE="~50MB" ;;
                numpy) SIZE_ESTIMATE="~20MB" ;;
                scipy) SIZE_ESTIMATE="~50MB" ;;
                matplotlib) SIZE_ESTIMATE="~40MB" ;;
                pillow) SIZE_ESTIMATE="~8MB" ;;
                scikit-image) SIZE_ESTIMATE="~30MB" ;;
                open3d) SIZE_ESTIMATE="~100MB" ;;
                transformers) SIZE_ESTIMATE="~200MB" ;;
            esac
            print_info "  ✓ $dep ($SIZE_ESTIMATE) - Essential for UAP analysis"
            HEAVY_FOUND=$((HEAVY_FOUND + 1))
        fi
    done
    
    if [ $HEAVY_FOUND -eq 0 ]; then
        print_success "✓ No heavy ML dependencies currently active"
        print_info "Commented dependencies available for advanced features"
    else
        print_info "Total heavy dependencies: $HEAVY_FOUND"
    fi
    
    # Check for optional/commented dependencies
    echo ""
    COMMENTED_DEPS=$(grep "^#.*torch\|^#.*tensorflow\|^#.*transformers\|^#.*open3d" requirements.txt | wc -l)
    if [ $COMMENTED_DEPS -gt 0 ]; then
        print_info "Optional advanced dependencies (commented): $COMMENTED_DEPS"
        print_info "Uncomment for ML-based UAP analysis features"
    fi
    
else
    print_warning "No requirements.txt found"
fi

# Check platform-specific requirements
echo ""
for platform_req in requirements-macos.txt requirements-linux.txt requirements-dev.txt build_requirements.txt; do
    if [ -f "$platform_req" ]; then
        COUNT=$(grep -v "^#" "$platform_req" | grep -v "^$" | wc -l)
        print_info "Platform requirements ($platform_req): $COUNT dependencies"
    fi
done

# 3. UAP Analysis Source code analysis
print_header "💻 UAP ANALYSIS SOURCE CODE"

# Analyze main source directories
if [ -d "src" ]; then
    print_info "UAP Analysis core modules:"
    for module in analyzers processors visualizers utils; do
        if [ -d "src/$module" ]; then
            MODULE_FILES=$(find "src/$module" -name "*.py" 2>/dev/null | wc -l)
            MODULE_LINES=$(find "src/$module" -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
            print_info "  $module/: $MODULE_FILES files ($MODULE_LINES lines)"
        fi
    done
    
    TOTAL_SRC_FILES=$(find src -name "*.py" | wc -l)
    TOTAL_SRC_LINES=$(find src -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
    print_success "Total src/: $TOTAL_SRC_FILES files ($TOTAL_SRC_LINES lines)"
fi

# Analyze scripts directory
if [ -d "scripts" ]; then
    SCRIPT_FILES=$(find scripts -name "*.py" | wc -l)
    SCRIPT_LINES=$(find scripts -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
    print_info "UAP GUI scripts/: $SCRIPT_FILES files ($SCRIPT_LINES lines)"
    
    echo ""
    print_info "Main UAP application scripts:"
    for script in uap_gui.py uap_analyzer_gui.py stable_gui.py run_analysis.py main_analyzer.py; do
        if [ -f "scripts/$script" ]; then
            LINES=$(wc -l < "scripts/$script" 2>/dev/null || echo "0")
            print_info "  $script: $LINES lines"
        fi
    done
fi

# Overall codebase statistics
echo ""
if [ -d "src" ] || [ -d "scripts" ]; then
    TOTAL_PY_FILES=$(find src scripts -name "*.py" 2>/dev/null | wc -l)
    TOTAL_PY_LINES=$(find src scripts -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
    print_success "Total UAP Analysis codebase: $TOTAL_PY_FILES files ($TOTAL_PY_LINES lines)"
fi

# 4. Assets and data analysis
print_header "📁 ASSETS & DATA ANALYSIS"

# Check for asset directories
for asset_dir in assets data configs examples docs; do
    if [ -d "$asset_dir" ]; then
        ASSET_SIZE=$(du -sh "$asset_dir" 2>/dev/null | cut -f1)
        ASSET_FILES=$(find "$asset_dir" -type f | wc -l)
        print_info "$asset_dir/: $ASSET_SIZE ($ASSET_FILES files)"
        
        # Check for large assets
        if [ "$asset_dir" = "assets" ] || [ "$asset_dir" = "data" ]; then
            echo "    Largest files:"
            find "$asset_dir" -type f -exec ls -lah {} + 2>/dev/null | sort -k5 -hr | head -3 | while read -r line; do
                SIZE=$(echo $line | awk '{print $5}')
                NAME=$(echo $line | awk '{print $9}')
                echo "      $SIZE - $(basename "$NAME")"
            done
        fi
    fi
done

# 5. Build output analysis
print_header "📦 BUILD OUTPUT ANALYSIS"

if [ -d "dist" ]; then
    DIST_SIZE=$(du -sb dist 2>/dev/null | cut -f1)
    DIST_SIZE_HR=$(human_readable $DIST_SIZE)
    print_info "Total dist size: $DIST_SIZE_HR"
    
    echo ""
    print_info "UAP Analysis build outputs:"
    
    # Check for UAP executables
    find dist -type f \( -name "UAP_Video_Analyzer*" -o -name "uap_*" -o -name "*.exe" -o -name "*.app" \) | while read -r file; do
        if [ -f "$file" ]; then
            SIZE=$(ls -lah "$file" | awk '{print $5}')
            NAME=$(basename "$file")
            
            # Convert size to MB for comparison
            SIZE_BYTES=$(ls -l "$file" | awk '{print $5}')
            SIZE_MB=$((SIZE_BYTES / 1024 / 1024))
            
            if [ $SIZE_MB -gt 200 ]; then
                print_warning "  ⚠️  $NAME: $SIZE (LARGE - expected for scientific computing)"
            elif [ $SIZE_MB -gt 100 ]; then
                print_info "  📦 $NAME: $SIZE (typical for UAP analysis)"
            else
                print_success "  ✓ $NAME: $SIZE"
            fi
        fi
    done
    
    # Check for scientific libraries in build
    echo ""
    print_info "Scientific libraries in UAP Analysis build:"
    SCIENCE_LIBS_FOUND=0
    for lib in cv2 numpy torch scipy matplotlib sklearn; do
        if find dist -name "*${lib}*" | head -1 | grep -q "$lib"; then
            print_info "  ✓ $lib library included"
            SCIENCE_LIBS_FOUND=$((SCIENCE_LIBS_FOUND + 1))
        fi
    done
    
    if [ $SCIENCE_LIBS_FOUND -gt 0 ]; then
        print_success "Scientific computing libraries: $SCIENCE_LIBS_FOUND (expected for UAP analysis)"
    fi
else
    print_warning "No dist directory found. Run a build first with ./compile-build-dist-python.sh"
fi

# 6. PyInstaller analysis
print_header "🔨 PYINSTALLER ANALYSIS"

if ls *.spec >/dev/null 2>&1; then
    SPEC_FILE=$(ls UAP_Video_Analyzer.spec 2>/dev/null || ls *.spec | head -1)
    print_info "PyInstaller spec file: $SPEC_FILE"
    
    # Analyze spec file
    if [ -f "$SPEC_FILE" ]; then
        if grep -q "onefile=True\|--onefile" "$SPEC_FILE"; then
            print_info "Build mode: Single file (--onefile)"
        else
            print_info "Build mode: Directory bundle"
        fi
        
        if grep -q "console=False\|--noconsole\|--windowed" "$SPEC_FILE"; then
            print_info "Console mode: Windowed (GUI mode)"
        else
            print_info "Console mode: Console visible"
        fi
        
        # Check for UAP-specific data files
        DATA_FILES=$(grep -c "datas=\|--add-data" "$SPEC_FILE" || echo "0")
        print_info "Additional data directories: $DATA_FILES"
        
        # Check for UAP hidden imports
        HIDDEN_IMPORTS=$(grep "hiddenimports=\|--hidden-import" "$SPEC_FILE" | wc -l)
        if [ $HIDDEN_IMPORTS -gt 0 ]; then
            print_info "Hidden imports specified: $HIDDEN_IMPORTS"
        fi
        
        # Check for UAP-specific collections
        if grep -q "collect-all" "$SPEC_FILE"; then
            print_info "Using collect-all for scientific libraries (expected)"
        fi
    fi
else
    print_warning "No PyInstaller spec file found"
    print_info "Will be generated on first build"
fi

# 7. UAP-specific optimization recommendations
print_header "💡 UAP ANALYSIS OPTIMIZATION RECOMMENDATIONS"

print_info "🔬 Scientific Computing Dependencies:"
if [ -d "venv" ] && [ $VENV_SIZE -gt 2147483648 ]; then
    print_info "  • Large environment expected for scientific computing"
    print_info "  • Consider CPU-only PyTorch if GPU not needed"
    print_info "  • Use opencv-python-headless if no GUI display needed"
fi

print_info "  • Current lightweight setup is optimal for basic analysis"
print_info "  • Uncomment advanced dependencies only when needed"
print_info "  • Consider lazy loading for ML models"

print_info "🏗️  UAP Analysis Build Optimization:"
print_info "  • Use --onefile for single executable distribution"
print_info "  • Use --windowed for clean GUI (default for UAP GUI)"
print_info "  • Enable UPX compression with --upx (reduces scientific lib size)"
print_info "  • Exclude test modules with --exclude-module"

print_info "📊 UAP Data & Assets:"
if [ -d "data" ]; then
    print_info "  • Consider external data loading for large datasets"
    print_info "  • Compress sample videos and reference data"
fi
if [ -d "assets" ]; then
    print_info "  • Optimize icon files for each platform"
    print_info "  • Use vector formats when possible"
fi

print_info "🎯 UAP-Specific Optimizations:"
print_info "  • Lazy load computer vision models"
print_info "  • Stream video processing instead of loading entire files"
print_info "  • Use progressive analysis for large video files"
print_info "  • Cache analysis results to avoid reprocessing"

# 8. UAP Analysis size targets
print_header "🎯 UAP ANALYSIS SIZE TARGETS & BENCHMARKS"

print_info "UAP Video Analyzer size guidelines:"
print_info "🔬 Scientific computing applications have different size expectations:"
echo ""
print_success "  ✓ Excellent (basic features): 50-150MB"
print_info "  📊 Good (with OpenCV): 150-300MB"
print_warning "  ⚠️  Acceptable (with ML): 300-800MB"
print_error "  ❌ Heavy (full ML suite): > 800MB"

echo ""
print_info "UAP Analysis quick optimization commands:"
echo "  # Dependency management"
echo "  pipreqs --force .  # Generate minimal requirements"
echo "  pip list --format=freeze | grep -v \"^-e\" > requirements-current.txt"
echo ""
echo "  # Optimized builds"
echo "  ./compile-build-dist-python.sh --onefile --windowed --upx"
echo "  ./compile-build-dist-python.sh --no-bloat-check  # Skip analysis for speed"
echo ""
echo "  # Development testing"
echo "  ./run-python-source.sh gui     # Test GUI from source"
echo "  ./run-python-source.sh console # Test analysis from source"
echo ""
echo "  # Cleanup"
echo "  find . -name '*.pyc' -delete && find . -name '__pycache__' -exec rm -rf {} +"

print_header "✅ UAP VIDEO ANALYZER BLOAT CHECK COMPLETE"

# Final recommendations based on UAP Analysis context
print_info "🛸 UAP Analysis maintenance tasks:"
print_info "  • Monitor scientific library versions for security updates"
print_info "  • Test video processing performance after dependency updates"
print_info "  • Keep sample data sets small for distribution"
print_info "  • Profile memory usage during video analysis"
print_info "  • Consider cloud processing for heavy ML inference"

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}  Ready to build optimized UAP Analysis application! 🛸${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"