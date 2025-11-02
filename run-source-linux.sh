#!/bin/bash
# Run UAP Analysis from source on Linux

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check for Python 3.8+
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Using Python: $PYTHON_VERSION"
else
    echo "Error: Python 3 not found"
    exit 1
fi

# Check for virtual environment
if [[ -d "$PROJECT_DIR/venv" ]]; then
    echo "Activating virtual environment..."
    source "$PROJECT_DIR/venv/bin/activate"
elif [[ -d "$PROJECT_DIR/.venv" ]]; then
    echo "Activating virtual environment..."
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Set Python path
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# Run the application
echo "Starting UAP Analysis GUI..."
python3 -m gui.stable_gui