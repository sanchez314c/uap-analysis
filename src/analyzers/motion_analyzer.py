#!/usr/bin/env python3
"""
Motion Analysis Component
========================

Analyzes motion patterns, trajectories, and movement characteristics in video frames.
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
import logging
import sys
from pathlib import Path

# Add utils to path for acceleration
sys.path.append(str(Path(__file__).parent.parent / "utils"))
try:
    from acceleration import get_acceleration_manager
except ImportError:
    # Fallback if acceleration module not available
    get_acceleration_manager = lambda config=None: None

logger = logging.getLogger(__name__)

class MotionAnalyzer:
    """Analyzes motion patterns and tracks object movement."""
    
    def __init__(self, config):
        """Initialize motion analyzer with configuration."""
        self.config = config
        self.motion_threshold = config['detection']['motion_threshold']
        self.optical_flow_params = config['detection']['optical_flow_params']
        
        # Initialize hardware acceleration
        self.accel_manager = None
        if config.get('acceleration', {}).get('auto_detect', True):
            self.accel_manager = get_acceleration_manager(config)
            if self.accel_manager:
                logger.info(f"Motion analysis using {self.accel_manager.device_type} acceleration")
        
    def analyze(self, frames, metadata):
        """Analyze motion patterns in video frames."""
        logger.info("Starting motion analysis...")
        
        if len(frames) < 2:
            logger.warning("Need at least 2 frames for motion analysis")
            return {}
        
        motion_vectors = []
        motion_energy = []
        frame_differences = []
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Setup progress bar if enabled
        if self.config['performance']['progress_bars']:
            frames_iter = tqdm(frames[1:], desc="Motion analysis")
        else:
            frames_iter = frames[1:]
        
        for frame in frames_iter:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow with hardware acceleration if available
            if (self.accel_manager and 
                self.config.get('acceleration', {}).get('accelerate_optical_flow', True)):
                flow = self.accel_manager.accelerate_optical_flow(
                    prev_gray, gray, **self.optical_flow_params
                )
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    **self.optical_flow_params
                )
            
            # Calculate motion magnitude
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = np.mean(magnitude)
            motion_energy.append(avg_magnitude)
            
            # Calculate frame difference
            diff = cv2.absdiff(prev_gray, gray)
            frame_differences.append(np.mean(diff))
            
            # Store motion vectors
            motion_vectors.append(flow)
            
            prev_gray = gray
        
        # Detect significant motion events
        significant_events = self._detect_motion_events(motion_energy, metadata)
        
        # Calculate trajectory
        trajectory = self._calculate_trajectory(motion_vectors)
        
        results = {
            'motion_energy': motion_energy,
            'frame_differences': frame_differences,
            'motion_vectors': motion_vectors,
            'significant_events': significant_events,
            'trajectory': trajectory,
            'average_motion': float(np.mean(motion_energy)) if motion_energy else 0.0,
            'max_motion': float(np.max(motion_energy)) if motion_energy else 0.0
        }
        
        logger.info(f"Motion analysis complete. {len(significant_events)} significant events detected.")
        return results
    
    def _detect_motion_events(self, motion_energy, metadata):
        """Detect significant motion events in the video."""
        if not motion_energy:
            return []
        
        # Calculate threshold for significant motion
        mean_energy = np.mean(motion_energy)
        std_energy = np.std(motion_energy)
        threshold = mean_energy + 2 * std_energy
        
        # Find peaks above threshold
        peaks, properties = find_peaks(
            motion_energy, 
            height=threshold,
            distance=metadata.get('fps', 30) // 4  # Minimum 0.25 seconds apart
        )
        
        # Convert to timestamp format
        fps = metadata.get('fps', 30)
        events = []
        for peak_idx in peaks:
            timestamp = (peak_idx + 1) / fps  # +1 because motion starts from frame 1
            intensity = motion_energy[peak_idx]
            events.append({
                'timestamp': timestamp,
                'frame': peak_idx + 1,
                'intensity': float(intensity)
            })
        
        return events
    
    def _calculate_trajectory(self, motion_vectors):
        """Calculate the overall trajectory of movement."""
        if not motion_vectors:
            return []
        
        # Calculate cumulative displacement
        trajectory = [(0, 0)]  # Start at origin
        
        for flow in motion_vectors:
            # Calculate average flow vector
            avg_flow_x = np.mean(flow[..., 0])
            avg_flow_y = np.mean(flow[..., 1])
            
            # Add to trajectory
            prev_x, prev_y = trajectory[-1]
            new_x = prev_x + avg_flow_x
            new_y = prev_y + avg_flow_y
            trajectory.append((float(new_x), float(new_y)))
        
        return trajectory
    
    def create_motion_visualization(self, frames, motion_vectors, output_path):
        """Create a visualization video of motion tracking."""
        if not motion_vectors:
            logger.warning("No motion vectors to visualize")
            return None
        
        # Setup video writer
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.config.get('video', {}).get('fps', 30)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Parameters for visualization
        step = 16  # Grid step for vector display
        
        # Create grid coordinates
        y_grid, x_grid = np.mgrid[step//2:height:step, step//2:width:step].reshape(2, -1).astype(int)
        
        # Add first frame (no motion data)
        out.write(frames[0])
        
        for i, (frame, flow) in enumerate(zip(frames[1:], motion_vectors)):
            # Copy frame for drawing
            vis_frame = frame.copy()
            
            # Extract flow vectors at grid points
            fx = flow[y_grid, x_grid, 0]
            fy = flow[y_grid, x_grid, 1]
            
            # Calculate magnitudes for color coding
            magnitudes = np.sqrt(fx*fx + fy*fy)
            max_magnitude = max(1, magnitudes.max())
            
            # Draw motion vectors
            for x, y, dx, dy, mag in zip(x_grid, y_grid, fx, fy, magnitudes):
                if mag < self.motion_threshold:
                    continue
                
                # Color based on magnitude (blue=slow, red=fast)
                color_intensity = min(255, int(255 * mag / max_magnitude))
                color = (255 - color_intensity, 0, color_intensity)
                
                # Draw arrow
                end_x = int(x + dx * 3)  # Scale for visibility
                end_y = int(y + dy * 3)
                cv2.arrowedLine(vis_frame, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
            
            # Add frame info
            cv2.putText(vis_frame, f"Frame: {i+1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(vis_frame)
        
        out.release()
        logger.info(f"Motion visualization saved to {output_path}")
        return str(output_path)