# Quick Start Guide

## Getting Started with UAP Analysis Suite

### Prerequisites

- Python 3.8 or higher
- Git for cloning the repository
- Sufficient disk space for video processing

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/uap-analysis.git
   cd uap-analysis
   ```

2. **Install dependencies**
   ```bash
   # Core dependencies
   pip install -r requirements.txt
   
   # Platform-specific dependencies
   pip install -r requirements-macos.txt    # macOS with Metal MPS
   pip install -r requirements-linux.txt    # Linux with CUDA/ROCm
   ```

3. **Verify installation**
   ```bash
   python scripts/test_setup.py
   ```

## Running the Application

### Option 1: GUI Application (Recommended)

```bash
python scripts/uap_gui.py
```

This launches the main graphical interface with full analysis capabilities.

### Option 2: Console Mode

For batch processing or command-line usage:

```bash
# Basic analysis
python scripts/run_analysis.py video.mp4

# Quick analysis (motion + luminosity + pulse only)
python scripts/run_analysis.py --quick video.mp4

# Advanced analysis with custom configuration
python scripts/run_advanced_analysis.py video.mp4 --config custom_config.yaml
```

### Option 3: Stable GUI Version

For production use:

```bash
python scripts/stable_gui.py
```

## First Analysis

1. **Launch the GUI** using one of the methods above
2. **Load your video file** through the file browser
3. **Configure analysis settings** (or use defaults)
4. **Start analysis** - the system will:
   - Extract and enhance video frames
   - Run parallel analysis across 15+ specialized modules
   - Generate comprehensive visualizations and reports
5. **Review results** in the output directory

## Output Locations

- **Enhanced Videos**: `results/enhanced/`
- **Analysis Data**: `results/analysis/`
- **Visualizations**: `results/visualizations/`
- **Reports**: `results/reports/`

## Common Tasks

### Building Applications

```bash
# Build standalone GUI application
python scripts/build.py --gui-only

# Build console application
python scripts/build.py --console-only

# Full build with distribution packages
python scripts/build.py --clean
```

### Testing

```bash
# Test project setup
python scripts/test_setup.py

# Test GUI functionality
python scripts/test_gui.py
```

## Getting Help

- **Documentation**: See `docs/` directory for detailed guides
- **Configuration**: Edit `configs/analysis_config.yaml`
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md`
- **Community**: Open an issue on GitHub

## Next Steps

1. Read the [Architecture Guide](ARCHITECTURE.md) to understand the system
2. Check the [Development Guide](DEVELOPMENT.md) for customization
3. Explore [Advanced Analysis Methods](ADVANCED_ANALYSIS_METHODS.md)
4. Review the [FAQ](FAQ.md) for common questions