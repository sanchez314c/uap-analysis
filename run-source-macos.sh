#!/bin/bash
# Run UAP Analysis from source on macOS

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if Python 3.11+ is available
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) -eq 1 ]]; then
        PYTHON_CMD="python3"
    else
        echo "Error: Python 3.11+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    echo "Error: Python 3 not found"
    exit 1
fi

# Activate conda environment if it exists
if command -v conda &> /dev/null; then
    if conda env list | grep -q "uap-gui"; then
        echo "Activating conda environment: uap-gui"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate uap-gui
        PYTHON_CMD="python"
    fi
fi

# Set Python path
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# Run the application
echo "Starting UAP Analysis GUI..."
$PYTHON_CMD -m gui.stable_gui