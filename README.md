# 🛸 UAP Analysis Suite

Advanced scientific analysis tool for Unidentified Aerial Phenomena (UAP) video investigation with machine learning and computer vision.

## ✅ Quick Start

### Prerequisites
- **Python 3.11+** (Required for proper GUI rendering on macOS)
- **OpenCV** for video processing
- **Tkinter** (included with Python)

### Installation

#### Option 1: Using Conda (Recommended for macOS)
```bash
# Create environment with Python 3.11
conda create -n uap-gui python=3.11 -y
conda activate uap-gui

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using venv
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### GUI Application
```bash
# macOS
./scripts/run-source-macos.sh

# Linux
./scripts/run-source-linux.sh

# Windows
scripts\run-source-windows.bat
```

#### Command Line Analysis
```bash
# Quick analysis
python src/run_advanced_analysis.py video.mp4 --quick

# Full analysis with all options
python src/run_advanced_analysis.py video.mp4 -o results --atmospheric --physics

# Individual analysis types
python src/run_advanced_analysis.py video.mp4 --atmospheric
python src/run_advanced_analysis.py video.mp4 --physics
```

## 📁 Repository Structure

```
uap-analysis2/
├── scripts/                       # Build and deployment scripts
│   ├── build-compile-dist.sh     # Universal build system
│   ├── run-source-macos.sh       # macOS launcher
│   ├── run-source-linux.sh       # Linux launcher
│   ├── run-source-windows.bat     # Windows launcher
│   ├── install-macos.sh          # macOS installer
│   ├── install-linux.sh          # Linux installer
│   └── [other build tools]        # Various utilities
├── src/                          # Source code
│   ├── gui/                      # GUI applications
│   │   └── stable_gui.py         # ✅ Main GUI application
│   ├── analyzers/                # Analysis engines
│   ├── run_advanced_analysis.py  # Command-line analysis tool
│   └── [other modules]           # Processing and visualization
├── configs/                      # Configuration files
├── archive/                      # Archived unused files
│   └── unused_gui_versions/      # Previous GUI versions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎮 GUI Features

The `stable_gui.py` provides a clean, stable interface that:
- ✅ Works reliably on macOS with Python 3.11
- ✅ Matches the interface of the compiled binary
- ✅ Supports Quick mode for faster analysis
- ✅ Includes Atmospheric and Physics analysis options
- ✅ Real-time progress logging
- ✅ Results folder integration

## 🔧 Analysis Options

### Quick Mode
- Faster analysis with core features
- Motion detection and basic tracking
- Essential luminosity analysis

### Advanced Analyses
- **Atmospheric Analysis**: Environmental modeling and interaction
- **Physics Analysis**: Trajectory validation and physics compliance
- **Additional options available via command line**

## 📊 Output

Analysis results are saved to:
- Enhanced video files with tracking overlays
- Motion tracking data (JSON format)
- Analysis reports (Markdown format)
- Detailed technical logs

## 🐛 Troubleshooting

### Gray Window Issues on macOS
If you experience gray windows:
1. **Use Python 3.11** (not 3.9 or earlier)
2. **Use conda environment** (recommended)
3. **Ensure tk/tkinter is properly installed**

```bash
# Verify Python version
python --version  # Should be 3.11.x

# Verify tkinter
python -c "import tkinter; print('✅ tkinter works')"
```

### Dependency Issues
```bash
# Install missing packages
pip install PyYAML opencv-python numpy matplotlib scipy tqdm Pillow

# Or install all requirements
pip install -r requirements.txt

```

## 📦 Building the Application

### Building a Self-Contained App

To create a standalone application that doesn't require Python installation:

```bash
# Using the universal build script (recommended)
./scripts/build-compile-dist.sh
```

The build script will:
1. Auto-detect the Python stack
2. Check for Python 3.11+ (required for macOS GUI compatibility)
3. Create a temporary build environment
4. Install all dependencies
5. Build a self-contained .app bundle using PyInstaller
6. Generate platform-specific run scripts

### Manual Build (Advanced)

If you prefer to build manually:

```bash
# 1. Create and activate a build environment with Python 3.11
conda create -n uap-build python=3.11 -y
conda activate uap-build

# 2. Install dependencies
pip install -r requirements.txt
pip install pyinstaller

# 3. Build the app
pyinstaller "UAP Video Analyzer.spec" --noconfirm --clean

# 4. Test the app
open "dist/UAP Video Analyzer.app"
```

### Important Notes

- **Python 3.11+ is required** for proper GUI rendering on macOS
- The build process creates a completely self-contained application
- No Python installation is needed to run the built app
- The app includes its own virtual environment with all dependencies

## 📝 Development Notes

### GUI Version History
- `stable_gui.py` - ✅ Current stable version (matches binary)
- Previous versions archived in `archive/unused_gui_versions/`
- Launch script updated to use stable version

### Key Fixes Applied
- Fixed Python path issues in analysis scripts
- Consolidated requirements files into unified structure
- Moved Python application files to src/ directory
- Created platform-specific run scripts (run-source-*)
- Identified Python 3.11 as required for macOS GUI compatibility
- Cleaned up unused GUI variants

## 🔗 Related Projects

- [OpenCV](https://opencv.org/) - Computer vision library
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Visualization

---

**Built with AI! 🤖**