# UAP Video Analyzer - Usage Instructions

## ✅ Recommended: Run From Source (Working Method)

### Quick Start
```bash
cd /Volumes/apfsRAID/Development/Github/UAP_Analysis
python scripts/uap_gui.py
```

### Why Source is Better
- ✅ **Works immediately** - All dependencies properly loaded
- ✅ **Full functionality** - No packaging limitations  
- ✅ **Easy to modify** - Update code and run instantly
- ✅ **Better performance** - No executable overhead

## 🔧 Alternative: Console Mode

For batch processing or command-line usage:
```bash
python scripts/run_analysis.py --help
```

## 📦 Built Executables Status

### macOS DMG (Packaging Issues)
- **Location**: `/dist/packages/UAP_Video_Analyzer_v2.0.0_macOS.dmg`
- **Status**: ⚠️ Dependency packaging incomplete
- **Issue**: PyInstaller failed to bundle scientific computing libraries
- **Recommendation**: Use source version instead

### Why the Build Has Issues
PyInstaller struggled with the complex dependency chain:
- NumPy, SciPy, OpenCV scientific computing stack
- PyTorch machine learning libraries  
- Multiple GUI framework dependencies
- Platform-specific binary libraries

## 🚀 Production Deployment Options

### Option 1: Python Environment Distribution (Recommended)
1. Package the source code with requirements.txt
2. Users install Python dependencies: `pip install -r requirements.txt`
3. Run from source: `python scripts/uap_gui.py`

### Option 2: Fix PyInstaller Build
- Manually specify all hidden imports
- Create custom hooks for scientific libraries
- Use virtual environment isolation
- Test extensively on clean systems

### Option 3: Docker Containerization
- Create Docker image with all dependencies
- Guaranteed consistent environment
- Cross-platform compatibility
- Easier distribution for technical users

## 💡 Current Status Summary

**✅ WORKING**: Source version with full functionality  
**⚠️ NEEDS WORK**: Standalone executables  
**📱 READY**: Professional GUI interface  
**🔬 COMPLETE**: Analysis algorithms and processing  

The application is **ready for use** via the source method!