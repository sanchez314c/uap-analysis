# Development Guide

## Overview

This guide provides comprehensive information for developers contributing to the UAP Analysis Suite. It covers development setup, coding standards, architecture understanding, and contribution workflows.

## Development Environment Setup

### Prerequisites

- **Python 3.8+** with development tools
- **Git** for version control
- **IDE**: VS Code, PyCharm, or similar
- **GPU**: NVIDIA/AMD/Apple Silicon for testing acceleration

### Initial Setup

```bash
# 1. Fork and clone repository
git clone https://github.com/your-username/UAP-Analysis.git
cd UAP-Analysis

# 2. Create development environment
python -m venv dev_env
source dev_env/bin/activate  # Linux/macOS
# or
dev_env\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Install in development mode
pip install -e .

# 5. Verify setup
python scripts/test_setup.py
pytest tests/
```

### IDE Configuration

#### VS Code

**Recommended Extensions**
- Python
- Python Docstring Generator
- GitLens
- Docker
- Jupyter

**Workspace Settings (.vscode/settings.json)**
```json
{
    "python.defaultInterpreterPath": "./dev_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true
    }
}
```

#### PyCharm

**Settings**
- Project Interpreter: Use dev_env
- Code Style: Black, 88 character line length
- Type Checker: mypy
- Test Runner: pytest

## Project Architecture

### Directory Structure

```
UAP-Analysis/
├── src/                     # Source code
│   ├── analyzers/          # Analysis modules
│   │   ├── __init__.py
│   │   ├── base.py         # Base classes
│   │   ├── motion_analyzer.py
│   │   ├── atmospheric_analyzer.py
│   │   ├── physics_analyzer.py
│   │   ├── signature_analyzer.py
│   │   ├── ml_classifier.py
│   │   ├── dimensional_analyzer.py
│   │   └── trajectory_predictor.py
│   ├── processors/         # Video processing
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── frame_processor.py
│   │   └── enhanced_processor.py
│   ├── visualizers/        # Visualization
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── video_3d_visualizer.py
│   │   ├── luminance_mapper.py
│   │   └── pulse_visualizer.py
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── acceleration.py
│       ├── config.py
│       ├── exceptions.py
│       └── logging.py
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── performance/       # Performance tests
│   └── fixtures/         # Test data
├── configs/               # Configuration files
├── examples/              # Usage examples
├── scripts/               # Applications
├── docs/                  # Documentation
└── requirements/          # Dependencies
    ├── base.txt
    ├── dev.txt
    ├── macos.txt
    └── linux.txt
```

### Component Interaction

```
┌─────────────────────────────────────────────────┐
│              Applications                  │
│  ┌─────────────┐ ┌─────────────────┐ │
│  │   GUI Apps   │ │  CLI Tools     │ │
│  └─────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│              API Layer                    │
│         (Public Interfaces)                │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│            Core Modules                   │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│ │Analyzers│ │Processor│ │Visualizer│ │
│ └─────────┘ └─────────┘ └─────────┘ │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│           Utility Layer                   │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│ │  Config │ │  GPU    │ │  I/O    │ │
│ │ Manager │ │Manager  │ │ Manager  │ │
│ └─────────┘ └─────────┘ └─────────┘ │
└─────────────────────────────────────────────────┘
```

## Coding Standards

### Python Style Guide

#### Code Formatting
- **Black**: 88 character line length
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check style
flake8 src/ tests/
mypy src/
```

#### Naming Conventions

```python
# Classes: PascalCase
class MotionAnalyzer:
    pass

# Functions and variables: snake_case
def analyze_trajectory(trajectory_data):
    max_velocity = calculate_max_velocity(trajectory_data)
    return max_velocity

# Constants: UPPER_SNAKE_CASE
MAX_FRAME_SIZE = 1920 * 1080
DEFAULT_CONFIDENCE_THRESHOLD = 0.8

# Private methods: underscore prefix
def _internal_calculation(self):
    pass

# Modules: snake_case
# motion_analyzer.py
# frame_processor.py
```

#### Type Hints

```python
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

def process_frames(
    frames: List[np.ndarray],
    config: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict], np.ndarray]:
    """
    Process a list of frames with optional configuration.
    
    Args:
        frames: List of video frames as numpy arrays
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (analysis_results, processed_frames)
    """
    pass
```

#### Documentation Strings

```python
class AtmosphericAnalyzer:
    """
    Analyzes atmospheric conditions and their effects on UAP phenomena.
    
    This module processes video data to detect and quantify atmospheric
    disturbances caused by unidentified aerial phenomena, including
    heat distortion, air displacement, and pressure waves.
    
    Attributes:
        config (Dict): Configuration parameters for analysis
        gpu_enabled (bool): Whether GPU acceleration is active
        
    Example:
        >>> analyzer = AtmosphericAnalyzer(config={'sensitivity': 0.8})
        >>> results = analyzer.analyze(frames)
        >>> print(results['heat_distortion_score'])
        0.75
    """
    
    def analyze(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze frames for atmospheric anomalies.
        
        Args:
            frames: List of video frames to analyze
            
        Returns:
            Dictionary containing:
            - heat_distortion_score: Float indicating heat anomaly level
            - air_displacement: Measured air displacement in meters
            - pressure_wave_amplitude: Pressure wave amplitude
            
        Raises:
            ValueError: If frames list is empty
            ProcessingError: If analysis fails
        """
        pass
```

### Testing Standards

#### Test Structure

```python
# tests/unit/test_motion_analyzer.py
import pytest
import numpy as np
from src.analyzers.motion_analyzer import MotionAnalyzer

class TestMotionAnalyzer:
    """Test suite for MotionAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        config = {'detection_threshold': 0.5}
        return MotionAnalyzer(config)
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample frame for testing."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_init_with_config(self, analyzer):
        """Test analyzer initialization with configuration."""
        assert analyzer.config['detection_threshold'] == 0.5
        assert analyzer.gpu_enabled is True
    
    def test_detect_objects_empty_frame(self, analyzer, sample_frame):
        """Test object detection on empty frame."""
        objects = analyzer.detect_objects(sample_frame)
        assert isinstance(objects, list)
        assert len(objects) == 0
    
    def test_detect_objects_with_objects(self, analyzer):
        """Test object detection with synthetic objects."""
        # Create frame with objects
        frame_with_objects = self._create_frame_with_objects()
        objects = analyzer.detect_objects(frame_with_objects)
        assert len(objects) > 0
        assert all('bbox' in obj for obj in objects)
    
    def test_analyze_trajectory_invalid_input(self, analyzer):
        """Test trajectory analysis with invalid input."""
        with pytest.raises(ValueError):
            analyzer.analyze_trajectory([])
    
    @pytest.mark.parametrize("threshold", [0.1, 0.5, 0.9])
    def test_different_thresholds(self, analyzer, sample_frame, threshold):
        """Test analyzer with different detection thresholds."""
        analyzer.config['detection_threshold'] = threshold
        # Test behavior with different thresholds
        results = analyzer.detect_objects(sample_frame)
        assert isinstance(results, list)
```

#### Test Categories

1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark critical paths
4. **GPU Tests**: Verify hardware acceleration
5. **End-to-End Tests**: Complete workflow validation

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_motion_analyzer.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run performance tests
pytest tests/performance/ -v

# Run GPU tests only
pytest -m gpu
```

## Development Workflow

### Git Workflow

#### Branch Strategy

```
main                    # Production-ready code
├── develop            # Integration branch
├── feature/xyz        # Feature branches
├── bugfix/abc        # Bug fix branches
└── hotfix/123        # Critical fixes
```

#### Commit Guidelines

```bash
# Feature branch
git checkout -b feature/new-analyzer

# Make changes
# ... (development work)

# Stage and commit
git add .
git commit -m """
feat: Add spectral analyzer module

- Implement SpectralAnalyzer class with multi-band analysis
- Add support for infrared, thermal, and UV spectrum
- Include GPU acceleration for FFT operations
- Add comprehensive unit tests

Closes #123
"""

# Push and create PR
git push origin feature/new-analyzer
```

#### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Examples**:
```
feat(analyzers): Add multi-spectral analysis capability

Implement new SpectralAnalyzer class that can process multiple
electromagnetic spectrum bands simultaneously. Includes GPU
acceleration for improved performance.

Closes #156

fix(processors): Handle corrupted video frames gracefully

Add error handling for corrupted frames that previously caused
crashes. Now logs warning and skips to next frame.

Fixes #201
```

### Code Review Process

#### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance benchmarks run

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Tests added/updated
```

#### Review Guidelines

1. **Functionality**: Does it work as intended?
2. **Performance**: Any performance implications?
3. **Security**: Any security concerns?
4. **Testing**: Are tests comprehensive?
5. **Documentation**: Is documentation clear?

## Debugging and Profiling

### Debugging Tools

#### Python Debugger

```python
# Using pdb
import pdb; pdb.set_trace()

# Using ipdb (recommended)
import ipdb; ipdb.set_trace()

# In VS Code, use breakpoints
```

#### Logging

```python
import logging
from src.utils import get_logger

logger = get_logger(__name__)

def analyze_frame(frame):
    logger.debug(f"Processing frame of shape: {frame.shape}")
    
    try:
        result = complex_analysis(frame)
        logger.info(f"Analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise
```

#### Performance Profiling

```python
# Line profiler
from line_profiler import LineProfiler

lp = LineProfiler()
lp_wrapper = lp(analyze_frame)
lp_wrapper(frame)
lp.print_stats()

# Memory profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function implementation
    pass
```

### GPU Debugging

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Memory usage
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

## Adding New Features

### Creating New Analyzer

1. **Create Base Class**
```python
# src/analyzers/base_analyzer.py
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    """Base class for all analyzers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gpu_enabled = config.get('gpu_enabled', False)
    
    @abstractmethod
    def analyze(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze frames and return results."""
        pass
    
    def validate_config(self) -> bool:
        """Validate analyzer configuration."""
        return True
```

2. **Implement Analyzer**
```python
# src/analyzers/new_analyzer.py
from .base_analyzer import BaseAnalyzer

class NewAnalyzer(BaseAnalyzer):
    """Custom analyzer for specific analysis type."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.setup_gpu()
    
    def analyze(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Main analysis implementation."""
        results = {}
        
        for i, frame in enumerate(frames):
            # Analysis logic here
            frame_result = self._process_frame(frame)
            results[f'frame_{i}'] = frame_result
        
        return self._aggregate_results(results)
    
    def _process_frame(self, frame: np.ndarray) -> Dict:
        """Process individual frame."""
        # Implementation
        pass
    
    def _aggregate_results(self, results: Dict) -> Dict:
        """Aggregate frame results."""
        # Implementation
        pass
```

3. **Add Tests**
```python
# tests/unit/test_new_analyzer.py
import pytest
from src.analyzers.new_analyzer import NewAnalyzer

class TestNewAnalyzer:
    def test_analyzer_initialization(self):
        config = {'param1': 'value1'}
        analyzer = NewAnalyzer(config)
        assert analyzer.config == config
    
    def test_analyze_empty_frames(self):
        analyzer = NewAnalyzer({})
        with pytest.raises(ValueError):
            analyzer.analyze([])
```

4. **Update Configuration**
```yaml
# configs/analysis_config.yaml
analysis_parameters:
  new_analysis:
    enabled: true
    sensitivity: 0.8
    output_format: "detailed"
```

5. **Register Analyzer**
```python
# src/analyzers/__init__.py
from .new_analyzer import NewAnalyzer

__all__ = [
    'MotionAnalyzer',
    'AtmosphericAnalyzer',
    'NewAnalyzer',
    # ... other analyzers
]
```

### Adding New Visualization

1. **Create Visualizer Class**
```python
# src/visualizers/new_visualizer.py
import matplotlib.pyplot as plt
from .base_visualizer import BaseVisualizer

class NewVisualizer(BaseVisualizer):
    """Custom visualization for new analysis type."""
    
    def create_visualization(self, data: Dict, output_path: str):
        """Create visualization from analysis data."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Visualization logic
        self._plot_data(ax, data)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
```

2. **Integrate with GUI**
```python
# scripts/uap_gui.py
from src.visualizers.new_visualizer import NewVisualizer

# Add to visualization options
visualization_options = {
    'trajectory': Video3DVisualizer,
    'luminance': LuminanceMapper,
    'new_visualization': NewVisualizer,
}
```

## Performance Optimization

### GPU Optimization

```python
# Use PyTorch for GPU operations
import torch

def gpu_accelerated_operation(data):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        tensor_data = torch.from_numpy(data).to(device)
        
        # GPU computation
        result = torch.fft.fft(tensor_data)
        
        return result.cpu().numpy()
    else:
        # CPU fallback
        return np.fft.fft(data)
```

### Memory Optimization

```python
# Process in chunks to reduce memory usage
def process_large_video(video_path, chunk_size=1000):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        if not frames:
            break
        
        # Process chunk
        yield process_chunk(frames)
        
        # Clear memory
        del frames
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def parallel_analysis(frames, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(analyze_frame, frame) for frame in frames]
        results = [f.result() for f in futures]
    
    return results
```

## Documentation

### Writing Documentation

1. **API Documentation**: Update docstrings
2. **User Guide**: Update relevant .md files
3. **Examples**: Add to examples/ directory
4. **Architecture**: Update ARCHITECTURE.md

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep documentation up-to-date

## Release Process

### Version Management

```bash
# Update version
bump2version patch  # 2.0.0 -> 2.0.1
bump2version minor  # 2.0.1 -> 2.1.0
bump2version major  # 2.1.0 -> 3.0.0
```

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Tagged in git
- [ ] Built and tested
- [ ] Release notes prepared

---

For additional development resources, see the [API documentation](API.md) and [architecture guide](ARCHITECTURE.md).