# API Documentation

## Overview

The UAP Analysis Suite provides a comprehensive Python API for programmatic access to all analysis capabilities. This document outlines the main interfaces and usage patterns.

## Core API Structure

### Analyzers Module

```python
from src.analyzers import (
    MotionAnalyzer,
    AtmosphericAnalyzer,
    PhysicsAnalyzer,
    SignatureAnalyzer,
    MLClassifier,
    DimensionalAnalyzer,
    TrajectoryPredictor
)
```

#### MotionAnalyzer

```python
class MotionAnalyzer:
    def __init__(self, config: dict = None, gpu_acceleration: bool = True):
        """Initialize motion analyzer with configuration."""
        
    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame."""
        
    def track_objects(self, frames: List[np.ndarray]) -> List[Track]:
        """Track objects across frames."""
        
    def analyze_trajectory(self, track: Track) -> TrajectoryAnalysis:
        """Analyze object trajectory."""
```

#### AtmosphericAnalyzer

```python
class AtmosphericAnalyzer:
    def __init__(self, config: dict = None):
        """Initialize atmospheric analyzer."""
        
    def parse_metadata(self, metadata: dict) -> AtmosphericConditions:
        """Parse environmental metadata."""
        
    def estimate_conditions(self, video_capture) -> AtmosphericConditions:
        """Estimate conditions from video."""
```

### Processors Module

```python
from src.processors import (
    FrameProcessor,
    EnhancedProcessor
)
```

#### FrameProcessor

```python
class FrameProcessor:
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance single frame quality."""
        
    def stabilize_frame(self, frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Stabilize frame using reference."""
```

### Visualizers Module

```python
from src.visualizers import (
    Video3DVisualizer,
    LuminanceMapper,
    PulseVisualizer
)
```

#### Video3DVisualizer

```python
class Video3DVisualizer:
    def reconstruct_3d_path(self, signatures: List[Signature]) -> Path3D:
        """Reconstruct 3D trajectory from signatures."""
        
    def create_interactive_plot(self, trajectory: Path3D, output_file: str):
        """Create interactive 3D visualization."""
```

## Usage Examples

### Basic Analysis Pipeline

```python
from src.analyzers import MotionAnalyzer, PhysicsAnalyzer
from src.processors import EnhancedProcessor
import cv2

# Initialize components
motion_analyzer = MotionAnalyzer(gpu_acceleration=True)
physics_analyzer = PhysicsAnalyzer()
processor = EnhancedProcessor()

# Process video
cap = cv2.VideoCapture("video.mp4")
results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Enhance frame
    enhanced = processor.enhance_frame(frame)
    
    # Detect and analyze
    objects = motion_analyzer.detect_objects(enhanced)
    for obj in objects:
        physics_result = physics_analyzer.analyze_trajectory(obj)
        results.append(physics_result)
```

### Advanced Analysis with Configuration

```python
import yaml
from src.analyzers import SignatureAnalyzer

# Load configuration
with open('configs/analysis_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize with custom config
signature_analyzer = SignatureAnalyzer(
    spectral_bands=config['signature_analysis']['electromagnetic_bands'],
    gpu_device=config['gpu_acceleration']['device']
)

# Run analysis
results = signature_analyzer.analyze_video(
    "video.mp4",
    environmental_conditions=conditions
)
```

## Configuration API

### Loading Configuration

```python
from src.utils import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config('configs/analysis_config.yaml')

# Access configuration sections
gpu_config = config.get('gpu_acceleration', {})
analysis_config = config.get('analysis_parameters', {})
```

### Dynamic Configuration

```python
# Update configuration at runtime
config_manager.update_config('analysis_parameters.motion_detection.sensitivity', 0.8)
config_manager.save_config('updated_config.yaml')
```

## GPU Acceleration API

### Acceleration Manager

```python
from src.utils import AccelerationManager

accel_manager = AccelerationManager()

# Check available acceleration
device_info = accel_manager.get_device_info()
print(f"Using device: {device_info['type']}")

# Configure GPU usage
accel_manager.configure(
    device_type='cuda',
    memory_fraction=0.8,
    mixed_precision=True
)
```

## Data Export API

### Results Export

```python
from src.utils import ExportManager

export_manager = ExportManager()

# Export to different formats
export_manager.export_json(results, 'analysis_results.json')
export_manager.export_csv(results, 'analysis_data.csv')
export_manager.export_pdf_report(results, 'analysis_report.pdf')
```

### Visualization Export

```python
# Export visualizations
export_manager.export_plot(trajectory_data, 'trajectory_plot.png')
export_manager.export_3d_model(3d_reconstruction, 'object_model.obj')
export_manager.export_video(enhanced_frames, 'enhanced_video.mp4')
```

## Error Handling

### Exception Classes

```python
from src.exceptions import (
    UAPAnalysisError,
    ConfigurationError,
    GPUAccelerationError,
    VideoProcessingError
)

try:
    analyzer = MotionAnalyzer(config)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except GPUAccelerationError as e:
    print(f"GPU acceleration error: {e}")
```

## Batch Processing API

### Batch Analyzer

```python
from src.utils import BatchProcessor

batch_processor = BatchProcessor()

# Process multiple videos
video_list = ['video1.mp4', 'video2.mp4', 'video3.mp4']
results = batch_processor.process_videos(
    video_list,
    analysis_config=config,
    parallel=True,
    max_workers=4
)

# Get summary report
summary = batch_processor.generate_summary(results)
```

## Integration Examples

### Jupyter Notebook Integration

```python
# In Jupyter notebooks
%matplotlib inline
from src.visualizers import InteractivePlotter

plotter = InteractivePlotter()
plotter.display_analysis_results(results)
```

### External Tool Integration

```python
# Integrate with external databases
from src.analyzers import DatabaseMatcher

db_matcher = DatabaseMatcher(
    databases=['aircraft.db', 'phenomena.db']
)

matches = db_matcher.find_matches(analysis_results)
```

## Performance Monitoring

### Performance Profiler

```python
from src.utils import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile('motion_analysis'):
    results = motion_analyzer.analyze_video(video)

# Get performance metrics
metrics = profiler.get_metrics()
print(f"Processing time: {metrics['motion_analysis']['duration']}s")
print(f"Memory usage: {metrics['motion_analysis']['peak_memory']}MB")
```

## API Versioning

The API follows semantic versioning. Current version: 2.0.0

- **Major versions**: Breaking changes
- **Minor versions**: New features, backward compatible
- **Patch versions**: Bug fixes, improvements

## Type Hints

All API functions include comprehensive type hints:

```python
from typing import List, Dict, Optional, Union
import numpy as np

def analyze_frame(
    frame: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """Analyze frame with optional configuration."""
    pass
```

## Async API

For async processing:

```python
import asyncio
from src.analyzers import AsyncAnalyzer

async def async_analysis():
    analyzer = AsyncAnalyzer()
    results = await analyzer.analyze_video_async('video.mp4')
    return results

# Run async analysis
results = asyncio.run(async_analysis())
```

## Plugin API

### Custom Analyzer Plugin

```python
from src.analyzers.base import BaseAnalyzer

class CustomAnalyzer(BaseAnalyzer):
    def __init__(self, config: dict):
        super().__init__(config)
        
    def analyze(self, frames: List[np.ndarray]) -> Dict:
        """Custom analysis implementation."""
        return {'custom_results': ...}

# Register plugin
from src.utils import PluginManager
plugin_manager = PluginManager()
plugin_manager.register_analyzer('custom', CustomAnalyzer)
```

---

For more detailed examples, see the [examples/](../examples/) directory and the [QUICK_START.md](QUICK_START.md) guide.