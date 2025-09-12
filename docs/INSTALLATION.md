# 🛠️ Installation Guide

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10

### Recommended for Optimal Performance
- **GPU**: NVIDIA GTX 1060+ or Apple Silicon M1+
- **RAM**: 32GB for large video files
- **Storage**: SSD with 50GB+ free space
- **CPU**: 8+ core processor with AVX support

## Quick Installation

### 🍎 macOS (with Metal Performance Shaders)
```bash
# Clone the repository
git clone https://github.com/yourusername/UAPAnalysis.git
cd UAPAnalysis

# Run automated installer
chmod +x install-macos.sh
./install-macos.sh

# Activate environment
source venv/bin/activate

# Test installation
python stable_gui.py
```

### 🐧 Linux (with CUDA support)
```bash
# Clone the repository
git clone https://github.com/yourusername/UAPAnalysis.git
cd UAPAnalysis

# Run automated installer
chmod +x install-linux.sh
./install-linux.sh

# Activate environment
source venv/bin/activate

# Test installation
python stable_gui.py
```

### 🪟 Windows
```bash
# Clone the repository
git clone https://github.com/yourusername/UAPAnalysis.git
cd UAPAnalysis

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python stable_gui.py
```

## Manual Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/UAPAnalysis.git
cd UAPAnalysis
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

#### Core Dependencies
```bash
pip install -r requirements.txt
```

#### Platform-Specific Acceleration

**macOS with Apple Silicon:**
```bash
pip install -r requirements-macos.txt
```

**Linux with NVIDIA GPU:**
```bash
pip install -r requirements-linux.txt
```

**Development Dependencies:**
```bash
pip install -r requirements-dev.txt
```

## Verification

### Test Basic Installation
```bash
# Test Python imports
python -c "import cv2, numpy, scipy, yaml; print('✅ All dependencies imported successfully')"

# Test GUI
python stable_gui.py
```

### Test Hardware Acceleration

**Test Metal Performance Shaders (macOS):**
```bash
python -c "import torch; print('✅ MPS available:', torch.backends.mps.is_available())"
```

**Test CUDA (Linux/Windows):**
```bash
python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available())"
```

### Test Analysis Pipeline
```bash
# Quick analysis test (if you have a test video)
python run_advanced_analysis.py test_video.mp4 --quick
```

## Troubleshooting

### Common Issues

#### "No module named cv2"
```bash
pip install opencv-python
```

#### "tkinter not found" (Linux)
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL
sudo yum install tkinter
```

#### CUDA Issues (Linux)
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA toolkit if needed
# Follow: https://developer.nvidia.com/cuda-downloads
```

#### Metal Performance Shaders Issues (macOS)
```bash
# Update to latest macOS
# Ensure using Python 3.9+ for best MPS support
python --version
```

### Performance Issues

#### Large Video Files
- Ensure sufficient RAM (32GB+ recommended)
- Use SSD storage for temporary files
- Enable hardware acceleration

#### Slow Processing
- Check GPU acceleration is working
- Reduce video resolution if needed
- Use Quick Mode for faster analysis

## Development Setup

### Additional Development Tools
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### GPU Development Setup

**NVIDIA Development:**
```bash
# Install CUDA development tools
pip install cupy-cuda11x  # or appropriate version
```

**Apple Silicon Development:**
```bash
# Install Metal development tools (automatic with PyTorch)
pip install torch torchvision torchaudio
```

## Environment Variables

### Optional Configuration
```bash
# Set custom data directory
export UAP_DATA_DIR="/path/to/data"

# Enable debug logging
export UAP_DEBUG=1

# Force CPU-only processing
export UAP_NO_GPU=1
```

## Updating

### Update from Git
```bash
git pull origin main
pip install -r requirements.txt
```

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## Uninstallation

### Remove Virtual Environment
```bash
# Deactivate if active
deactivate

# Remove virtual environment
rm -rf venv/

# Remove project directory
cd ..
rm -rf UAPAnalysis/
```

---

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/yourusername/UAPAnalysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/UAPAnalysis/discussions)
- **Documentation**: [Project Wiki](https://github.com/yourusername/UAPAnalysis/wiki)