#!/bin/bash
# UAP Analysis Suite - Quick Build Script for Unix-like systems (macOS/Linux)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status "Using Python $PYTHON_VERSION"
    
    # Check minimum Python version (3.8)
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_error "Python 3.8 or higher is required"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing build dependencies..."
    
    if [ -f "requirements.txt" ]; then
        python3 -m pip install -r requirements.txt
    fi
    
    if [ -f "build_requirements.txt" ]; then
        python3 -m pip install -r build_requirements.txt
    fi
    
    # Install platform-specific requirements
    case "$(uname)" in
        "Darwin")
            if [ -f "requirements-macos.txt" ]; then
                python3 -m pip install -r requirements-macos.txt
            fi
            ;;
        "Linux")
            if [ -f "requirements-linux.txt" ]; then
                python3 -m pip install -r requirements-linux.txt
            fi
            ;;
    esac
    
    print_success "Dependencies installed"
}

# Main build function
build_application() {
    print_status "Starting UAP Analysis Suite build..."
    
    # Parse command line arguments
    CLEAN_BUILD=false
    VERBOSE=false
    PLATFORMS="current"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --platforms)
                PLATFORMS="$2"
                shift
                shift
                ;;
            --help|-h)
                echo "UAP Analysis Suite Build Script"
                echo ""
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --clean      Clean previous build artifacts"
                echo "  --verbose    Enable verbose output"
                echo "  --platforms  Platforms to build for (current,linux,windows,macos,all)"
                echo "  --help, -h   Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                           # Quick build for current platform"
                echo "  $0 --clean --verbose        # Clean build with verbose output"
                echo "  $0 --platforms all          # Build for all platforms"
                echo ""
                exit 0
                ;;
            *)
                print_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Build command arguments
    BUILD_ARGS=""
    if [ "$CLEAN_BUILD" = true ]; then
        BUILD_ARGS="$BUILD_ARGS --clean"
    fi
    if [ "$VERBOSE" = true ]; then
        BUILD_ARGS="$BUILD_ARGS --verbose"
    fi
    if [ "$PLATFORMS" != "current" ]; then
        BUILD_ARGS="$BUILD_ARGS --platforms $PLATFORMS"
    fi
    
    print_status "Build configuration:"
    print_status "  Platforms: $PLATFORMS"
    print_status "  Clean build: $CLEAN_BUILD"
    print_status "  Verbose: $VERBOSE"
    
    # Run the build
    python3 build_all.py $BUILD_ARGS
    
    if [ $? -eq 0 ]; then
        print_success "Build completed successfully!"
        print_status "Check build-compile-dist/packages/ for output files"
    else
        print_error "Build failed!"
        exit 1
    fi
}

# Main execution
main() {
    echo "🛸 UAP Analysis Suite - Build Script"
    echo "=================================="
    
    check_python
    
    # Ask user if they want to install dependencies
    read -p "Install/update build dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_dependencies
    fi
    
    build_application "$@"
    
    print_success "All done! 🚀"
}

# Run main function with all arguments
main "$@"