# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**UAP Analysis Suite** is a comprehensive scientific video analysis system designed for investigating Unidentified Aerial Phenomena (UAP). The project uses advanced computer vision, machine learning, and physics-based modeling to analyze aerial video footage with GPU acceleration across multiple platforms.

## Core Commands

### Environment Setup
```bash
# Install core dependencies
pip install -r requirements.txt

# Platform-specific dependencies
pip install -r requirements-macos.txt    # macOS with Metal MPS
pip install -r requirements-linux.txt    # Linux with CUDA/ROCm

# Test installation and setup
python scripts/test_setup.py
```

### Building Applications
```bash
# Build standalone GUI application
python scripts/build.py --gui-only

# Build console application
python scripts/build.py --console-only

# Full build (both GUI and console with distribution packages)
python scripts/build.py --clean

# Platform-specific builds
python scripts/build_macos.py            # macOS application
python scripts/build_windows.py          # Windows executable
python scripts/build_all.py              # All platforms
```

### Running Analysis
```bash
# Basic video analysis
python scripts/run_analysis.py video.mp4

# Quick analysis (motion + luminosity + pulse only)
python scripts/run_analysis.py --quick video.mp4

# Advanced analysis with custom configuration
python scripts/run_advanced_analysis.py video.mp4 --config custom_config.yaml

# GUI applications
python scripts/uap_gui.py                # Main GUI
python scripts/stable_gui.py             # Stable GUI version
```

### Testing and Validation
```bash
# Test project setup and structure
python scripts/test_setup.py

# Test GUI functionality
python scripts/test_gui.py
```

## Architecture Overview

### Core Module Structure
```
src/
├── analyzers/          # Analysis engines (15+ specialized analyzers)
│   ├── legacy_comprehensive_analyzer.py    # Main UAP analysis class
│   ├── motion_analyzer.py                  # Object tracking and physics
│   ├── atmospheric_analyzer.py             # Environmental modeling
│   ├── physics_analyzer.py                 # Trajectory validation
│   ├── signature_analyzer.py               # EM/thermal signatures
│   ├── ml_classifier.py                    # Machine learning classification
│   ├── dimensional_analyzer.py             # Size/distance estimation
│   └── trajectory_predictor.py             # Behavioral prediction
├── processors/        # Video processing pipeline
│   ├── frame_processor.py                 # Individual frame enhancement
│   └── enhanced_processor.py              # Multi-frame processing
├── visualizers/       # 3D visualization and rendering
│   ├── video_3d_visualizer.py             # 3D trajectory visualization
│   ├── luminance_mapper.py                # Light pattern analysis
│   └── pulse_visualizer.py                # Temporal pattern display
└── utils/             # Utility functions and GPU acceleration
    └── acceleration.py                    # GPU acceleration helpers
```

### Analysis Pipeline Flow
1. **Input Processing**: Video extraction and frame enhancement
2. **Multi-Modal Analysis**: Parallel analysis across 15+ specialized modules
3. **Data Fusion**: Cross-module correlation and pattern detection
4. **Output Generation**: Enhanced videos, visualizations, and scientific reports
5. **Validation**: Physics-based verification and anomaly scoring

### Key Design Patterns
- **Modular Architecture**: Each analyzer operates independently with standardized interfaces
- **GPU Acceleration**: Automatic detection and use of CUDA/ROCm/Metal where available
- **Configuration-Driven**: YAML-based configuration controls analysis parameters and outputs
- **Multi-Format Support**: Handles various video formats with automatic preprocessing

## Configuration Management

### Main Configuration: `configs/analysis_config.yaml`
Controls analysis types, output formats, GPU settings, and performance parameters. The system validates all configurations before processing.

### Dynamic Configuration
Analysis can be modified at runtime through command-line arguments and custom config files.

## GPU Acceleration Support

The project automatically detects and uses available GPU acceleration:
- **macOS**: Metal Performance Shaders (MPS) on Apple Silicon
- **Linux/Windows**: CUDA 11.0+ on NVIDIA GPUs
- **AMD Systems**: ROCm 5.0+ on AMD GPUs
- **Fallback**: CPU-based processing when GPU unavailable

## Output Structure

Analysis generates comprehensive outputs including:
- **Enhanced Videos**: Motion tracking, stabilization, EM simulation
- **Analysis Data**: Motion vectors, luminosity patterns, spectral analysis
- **Visualizations**: 3D trajectories, light patterns, temporal analysis
- **Scientific Reports**: Structured data and human-readable summaries

## Key Technical Notes

- Memory requirements scale with video resolution and analysis complexity
- GPU memory usage is configurable (default: 80% of available VRAM)
- The project maintains scientific rigor with validation and peer review capabilities
- All analysis includes chain-of-custody verification for scientific integrity

## Development Best Practices

- Use the provided test suite to verify setup: `python scripts/test_setup.py`
- Build with `--clean` flag for distribution packages
- GPU acceleration is transparent - no code changes needed for different platforms
- Configuration changes should be made through YAML files rather than code modifications