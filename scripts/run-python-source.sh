#!/bin/bash

# Run UAP Video Analyzer from Source (Development Mode)
# Launches the UAP analysis app directly from source code

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✔${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] ℹ${NC} $1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

show_help() {
    echo ""
    echo -e "${PURPLE}🛸 UAP Video Analyzer - Development Runner 🛸${NC}"
    echo ""
    echo "Usage: ./run-python-source.sh [options] [script]"
    echo ""
    echo "Scripts:"
    echo "  gui                Run main GUI application (default)"
    echo "  console            Run console analysis tool"
    echo "  stable             Run stable GUI version"
    echo "  advanced           Run advanced analysis"
    echo ""
    echo "Options:"
    echo "  --no-deps          Skip dependency check"
    echo "  --console          Force console output (for GUI debugging)"
    echo "  --help             Show this help"
    echo ""
    echo "Examples:"
    echo "  ./run-python-source.sh                # Run main UAP GUI"
    echo "  ./run-python-source.sh console        # Run console analysis"
    echo "  ./run-python-source.sh stable         # Run stable GUI"
    echo "  ./run-python-source.sh --console gui  # GUI with console output"
    echo ""
}

# Parse command line arguments
SKIP_DEPS=false
FORCE_CONSOLE=false
SCRIPT_TYPE="gui"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-deps)
            SKIP_DEPS=true
            shift
            ;;
        --console)
            FORCE_CONSOLE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        gui|console|stable|advanced)
            SCRIPT_TYPE="$1"
            shift
            ;;
        *)
            print_warning "Unknown option: $1"
            shift
            ;;
    esac
done

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}  🛸 UAP VIDEO ANALYZER - DEVELOPMENT MODE 🛸${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
echo ""

print_status "🚀 Starting UAP Video Analyzer from source..."
print_info "Mode: Development ($SCRIPT_TYPE)"

# Determine Python command
PYTHON_CMD=""
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    print_error "Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>/dev/null | cut -d' ' -f2)
print_info "Python version: $PYTHON_VERSION"

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    print_status "Activating UAP Analysis virtual environment..."
    if [ "$(uname)" = "Darwin" ] || [ "$(uname)" = "Linux" ]; then
        source venv/bin/activate
    else
        source venv/Scripts/activate
    fi
    print_success "Virtual environment activated"
else
    print_warning "No virtual environment found, using system Python"
    print_info "Recommendation: Create virtual environment with 'python3 -m venv venv'"
fi

# Install dependencies if needed and not skipped
if [ "$SKIP_DEPS" = false ]; then
    print_status "Checking UAP Analysis dependencies..."
    
    # Check for main dependencies
    MISSING_DEPS=()
    
    # Check core UAP Analysis dependencies
    for dep in opencv-python numpy matplotlib scipy tqdm PyYAML; do
        if ! $PYTHON_CMD -c "import ${dep//-/_}" 2>/dev/null; then
            MISSING_DEPS+=($dep)
        fi
    done
    
    # Check tkinter (usually built-in)
    if ! $PYTHON_CMD -c "import tkinter" 2>/dev/null; then
        print_warning "tkinter not available - GUI may not work"
    fi
    
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        print_warning "Missing dependencies: ${MISSING_DEPS[*]}"
        
        if [ -f "requirements.txt" ]; then
            print_status "Installing from requirements.txt..."
            pip install -r requirements.txt
            if [ $? -ne 0 ]; then
                print_error "Failed to install UAP Analysis dependencies"
                print_info "Try: pip install opencv-python numpy matplotlib scipy tqdm PyYAML"
                exit 1
            fi
            print_success "Dependencies installed"
        else
            print_error "requirements.txt not found"
            print_info "Please install manually: pip install opencv-python numpy matplotlib scipy tqdm PyYAML"
            exit 1
        fi
    else
        print_success "All UAP Analysis dependencies available"
    fi
fi

# Determine which script to run
MAIN_FILE=""
case $SCRIPT_TYPE in
    "gui")
        MAIN_FILE="scripts/uap_gui.py"
        print_info "Running: Main UAP GUI interface"
        ;;
    "console")
        MAIN_FILE="scripts/run_analysis.py"
        print_info "Running: Console analysis tool"
        FORCE_CONSOLE=true
        ;;
    "stable")
        MAIN_FILE="scripts/stable_gui.py"
        print_info "Running: Stable GUI version"
        ;;
    "advanced")
        MAIN_FILE="scripts/run_advanced_analysis.py"
        print_info "Running: Advanced analysis tool"
        ;;
esac

# Fallback options if main file doesn't exist
if [ ! -f "$MAIN_FILE" ]; then
    print_warning "Primary script not found: $MAIN_FILE"
    
    # Try alternative UAP Analysis scripts
    for candidate in scripts/uap_analyzer_gui.py scripts/main_analyzer.py scripts/uap_gui.py; do
        if [ -f "$candidate" ]; then
            MAIN_FILE="$candidate"
            print_info "Using alternative: $MAIN_FILE"
            break
        fi
    done
fi

if [ -z "$MAIN_FILE" ] || [ ! -f "$MAIN_FILE" ]; then
    print_error "Cannot find UAP Analysis main script"
    print_info "Expected locations:"
    print_info "  • scripts/uap_gui.py (main GUI)"
    print_info "  • scripts/run_analysis.py (console)"
    print_info "  • scripts/stable_gui.py (stable GUI)"
    print_info "  • scripts/run_advanced_analysis.py (advanced)"
    exit 1
fi

# Check if script imports look correct
if ! grep -q "import\|from" "$MAIN_FILE" 2>/dev/null; then
    print_warning "Script may be empty or invalid: $MAIN_FILE"
fi

# Set up environment for UAP Analysis
export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR:$PYTHONPATH"
print_info "Python path: $PYTHONPATH"

# Display run information
print_status "UAP Analysis execution details:"
print_info "  Script: $MAIN_FILE"
print_info "  Working directory: $SCRIPT_DIR"
print_info "  Console mode: $([ "$FORCE_CONSOLE" = true ] && echo "Enabled" || echo "Default")"

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
print_status "🛸 Launching UAP Video Analyzer..."
print_status "Press Ctrl+C to stop the application"
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
echo ""

# Run the UAP Analysis application
if [ "$FORCE_CONSOLE" = true ]; then
    # Force console output (useful for debugging GUI apps)
    $PYTHON_CMD "$MAIN_FILE" 2>&1
else
    # Normal execution
    $PYTHON_CMD "$MAIN_FILE"
fi

# Capture exit code
EXIT_CODE=$?

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    print_success "🛸 UAP Analysis session completed successfully"
elif [ $EXIT_CODE -eq 130 ]; then
    print_info "🛸 UAP Analysis session interrupted by user (Ctrl+C)"
else
    print_error "🛸 UAP Analysis session ended with errors (exit code: $EXIT_CODE)"
    
    # Helpful debugging info
    echo ""
    print_info "Debugging tips:"
    print_info "  • Run with --console flag to see error output"
    print_info "  • Check that all dependencies are installed"
    print_info "  • Verify that video files are in correct format"
    print_info "  • Check file permissions for input/output directories"
fi
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"