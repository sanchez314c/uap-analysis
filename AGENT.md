# Claude Instructions for UAP Video Analyzer

## Project Overview
UAP Video Analyzer is a Python application for analyzing Unidentified Aerial Phenomena (UAP) video footage using computer vision and data analysis techniques.

## Technology Stack
- **Language**: Python 3.8+
- **GUI**: Tkinter
- **Build System**: PyInstaller
- **Key Libraries**: OpenCV, NumPy, pandas

## Project Structure
```
uap-analysis/
├── src/                # Python source code
├── scripts/            # Utility and build scripts
├── config/             # Configuration files
├── data/               # Data files (raw, processed, results)
├── assets/             # Icons, images, resources
├── docs/               # Documentation
├── dev/                # Development resources
├── tests/              # Test suites
├── examples/           # Usage examples
└── results/            # Analysis outputs
```

## Common Tasks

### Development
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
python src/main.py
```

### Building
```bash
pyinstaller UAP\ Video\ Analyzer.spec
```

## Key Files
- `src/` - Main application code
- `requirements.txt` - Python dependencies
- `requirements-macos.txt` - macOS-specific dependencies
- `requirements-linux.txt` - Linux-specific dependencies
- `UAP Video Analyzer.spec` - PyInstaller build configuration

## Recent Changes
- Project standardized on October 8, 2025
- Created comprehensive docs structure
- Reorganized config files to config/
- Added dev/ directory for specifications
