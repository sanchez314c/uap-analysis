# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Python Version Compatibility
**Problem**: `SyntaxError` or import errors
**Solution**: 
- Ensure Python 3.8 or higher
- Use `python --version` to check
- Consider using conda or pyenv for version management

#### Dependency Installation Failures
**Problem**: `pip install` fails for scientific packages
**Solutions**:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev build-essential

# Use conda for scientific packages
conda install numpy scipy opencv pytorch

# For macOS with Apple Silicon
pip install --no-binary :all: numpy scipy
```

#### GPU Acceleration Issues
**Problem**: CUDA/Metal not detected
**Solutions**:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# For macOS Metal support
pip install torch torchvision torchaudio

# For Linux CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Runtime Issues

#### Memory Errors
**Problem**: `OutOfMemoryError` during analysis
**Solutions**:
- Reduce video resolution before processing
- Adjust GPU memory usage in `configs/analysis_config.yaml`
- Use `--quick` mode for initial analysis
- Close other applications using GPU memory

#### Video Processing Errors
**Problem**: Cannot read video file
**Solutions**:
- Check video format compatibility (MP4, AVI, MOV supported)
- Install FFmpeg for additional format support
- Verify file permissions and path
- Try converting video to MP4 format first

#### GUI Launch Failures
**Problem**: GUI window doesn't appear
**Solutions**:
```bash
# Check Tkinter installation
python -c "import tkinter; print('Tkinter OK')"

# For macOS XQuartz issues
brew install --cask xquartz

# Try alternative GUI
python scripts/stable_gui.py
```

### Performance Issues

#### Slow Processing
**Problem**: Analysis taking too long
**Solutions**:
- Enable GPU acceleration in config
- Reduce analysis modules in `configs/analysis_config.yaml`
- Use lower resolution video
- Close unnecessary background applications

#### High CPU Usage
**Problem**: System becomes unresponsive
**Solutions**:
- Limit concurrent processes in config
- Use `--quick` mode for initial analysis
- Schedule analysis for off-peak hours

### Build Issues

#### PyInstaller Failures
**Problem**: Cannot create standalone executable
**Solutions**:
- Use virtual environment: `python -m venv venv`
- Install all dependencies in clean environment
- Try `python scripts/build.py --clean`
- Consider Docker containerization instead

#### macOS Code Signing Errors
**Problem**: App cannot be opened due to security
**Solutions**:
```bash
# Allow app to run
xattr -rd com.apple.quarantine "UAP Video Analyzer.app"

# Or run from source instead
python scripts/uap_gui.py
```

### Data Issues

#### Corrupted Output Files
**Problem**: Analysis results are incomplete
**Solutions**:
- Check available disk space
- Verify video file integrity
- Run with `--debug` flag for detailed logs
- Try processing shorter video segment

#### Inconsistent Results
**Problem**: Same video gives different results
**Solutions**:
- Check random seed settings in config
- Ensure consistent GPU/CPU processing mode
- Verify video file hasn't changed
- Check for software version differences

### Network Issues

#### Download Failures
**Problem**: Cannot download models or dependencies
**Solutions**:
- Check internet connection
- Use VPN if behind firewall
- Set proxy in environment variables
- Download models manually and place in correct directory

## Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Console mode with debug
python scripts/run_analysis.py --debug video.mp4

# GUI with debug logging
DEBUG=1 python scripts/uap_gui.py
```

## Getting Help

### Collect Diagnostic Information
```bash
# System information
python scripts/test_setup.py > system_info.txt

# GPU information
nvidia-smi > gpu_info.txt  # NVIDIA
system_profiler SPDisplaysDataType > gpu_info.txt  # macOS

# Package versions
pip freeze > package_versions.txt
```

### Report Issues
When reporting issues, include:
1. Operating system and version
2. Python version
3. Error messages (full traceback)
4. Steps to reproduce
5. Diagnostic information from above

### Community Support
- GitHub Issues: Report bugs and request features
- Documentation: Check `docs/` for detailed guides
- FAQ: See [FAQ.md](FAQ.md) for common questions