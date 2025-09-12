#!/bin/bash
# UAP Analysis Installation Script for Linux
# ===========================================
# This script installs the UAP Analysis Pipeline with CUDA support

set -e  # Exit on any error

echo "🐧 UAP Analysis Pipeline - Linux Installation"
echo "=============================================="

# Check if Python 3.8+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required (found $python_version)"
    echo "   Install with: sudo apt update && sudo apt install python3.8 python3.8-venv python3.8-dev"
    exit 1
fi

echo "✅ Python $python_version detected"

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits | head -1
    CUDA_AVAILABLE=true
else
    echo "⚠️  NVIDIA driver not found - will use CPU acceleration"
    CUDA_AVAILABLE=false
fi

# Check for CUDA toolkit
if [ "$CUDA_AVAILABLE" = true ]; then
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
        echo "✅ CUDA toolkit detected: $cuda_version"
    else
        echo "⚠️  CUDA toolkit not found - install from https://developer.nvidia.com/cuda-downloads"
        echo "   Will attempt to install PyTorch with CUDA anyway"
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📥 Installing base dependencies..."
pip install --upgrade pip

# Install core requirements
pip install -r requirements.txt

# Install Linux-specific requirements
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🚀 Installing CUDA support..."
    # Try to install CUDA version
    pip install torch>=2.0.0+cu118 torchvision>=0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118 || {
        echo "⚠️  CUDA installation failed, falling back to CPU version"
        pip install torch>=2.0.0+cpu torchvision>=0.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
    }
else
    echo "📦 Installing CPU-only PyTorch..."
    pip install torch>=2.0.0+cpu torchvision>=0.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
fi

# Install additional Linux requirements
pip install -r requirements-linux.txt || echo "⚠️  Some optional dependencies failed to install"

# Verify CUDA support if available
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🧪 Testing CUDA support..."
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'✅ CUDA is available! Device: {torch.cuda.get_device_name(0)}')
    device = torch.device('cuda')
    x = torch.randn(100, 100, device=device)
    print('✅ CUDA test computation successful')
else:
    print('⚠️  CUDA not available, will use CPU acceleration')
" || echo "⚠️  PyTorch CUDA test failed"
fi

# Install system dependencies if needed
echo "📦 Checking system dependencies..."
if ! python3 -c "import cv2" &> /dev/null; then
    echo "Installing OpenCV system dependencies..."
    if command -v apt &> /dev/null; then
        sudo apt update
        sudo apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgtk-3-0
    elif command -v yum &> /dev/null; then
        sudo yum install -y glib2 libSM libXext libXrender libgomp gtk3
    fi
fi

# Test basic functionality
echo "🔬 Running setup test..."
python3 test_setup.py

# Make scripts executable
chmod +x run_analysis.py

echo ""
echo "🎉 Installation complete!"
echo "================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To analyze a video:"
echo "  python run_analysis.py your_video.mp4"
echo ""
echo "To run a quick analysis:"
echo "  python run_analysis.py --quick your_video.mp4"
echo ""
echo "Hardware acceleration status:"
python3 -c "
import sys; sys.path.insert(0, 'src/utils')
try:
    from acceleration import get_acceleration_manager
    accel = get_acceleration_manager()
    info = accel.get_device_info()
    print(f'  Device: {info[\"device_type\"]}')
    print(f'  Backend: {info[\"backend\"]}')
    if info['device_type'] == 'cuda':
        print(f'  🚀 CUDA enabled on {info.get(\"gpu_name\", \"GPU\")}!')
    elif info['device_type'] == 'opencl':
        print('  🚀 OpenCL acceleration enabled!')
except:
    print('  ⚠️  Using CPU acceleration')
"
echo ""
echo "For more information, see README.md"