#!/usr/bin/env python3
"""
Luminosity Analysis Component
============================

Analyzes light patterns, intensity changes, and luminosity characteristics.
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class LuminosityAnalyzer:
    """Analyzes luminosity patterns and light characteristics."""
    
    def __init__(self, config):
        """Initialize luminosity analyzer with configuration."""
        self.config = config
        
    def analyze(self, frames, metadata):
        """Analyze luminosity patterns in video frames."""
        logger.info("Starting luminosity analysis...")
        
        if not frames:
            logger.warning("No frames provided for luminosity analysis")
            return {}
        
        # Initialize data storage
        avg_luminosity = []
        max_luminosity = []
        min_luminosity = []
        center_luminosity = []
        
        # Define center region for focused analysis
        height, width = frames[0].shape[:2]
        center_h_start = height // 3
        center_h_end = (height // 3) * 2
        center_w_start = width // 3
        center_w_end = (width // 3) * 2
        
        # Setup progress bar if enabled
        if self.config['performance']['progress_bars']:
            frames_iter = tqdm(frames, desc="Luminosity analysis")
        else:
            frames_iter = frames
        
        for frame in frames_iter:
            # Convert to grayscale for luminosity analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate overall luminosity metrics
            avg_luminosity.append(np.mean(gray))
            max_luminosity.append(np.max(gray))
            min_luminosity.append(np.min(gray))
            
            # Calculate center region luminosity
            center_region = gray[center_h_start:center_h_end, center_w_start:center_w_end]
            center_luminosity.append(np.mean(center_region))
        
        # Create timestamps
        fps = metadata.get('fps', 30)
        timestamps = [i / fps for i in range(len(frames))]
        
        # Analyze for pulse patterns
        pulse_analysis = self._analyze_pulse_patterns(center_luminosity, fps)
        
        # Detect luminosity anomalies
        anomalies = self._detect_luminosity_anomalies(avg_luminosity, timestamps)
        
        results = {
            'timestamps': timestamps,
            'average_luminosity': avg_luminosity,
            'maximum_luminosity': max_luminosity,
            'minimum_luminosity': min_luminosity,
            'center_luminosity': center_luminosity,
            'pulse_analysis': pulse_analysis,
            'anomalies': anomalies,
            'statistics': {
                'mean_luminosity': float(np.mean(avg_luminosity)),
                'std_luminosity': float(np.std(avg_luminosity)),
                'luminosity_range': float(np.max(avg_luminosity) - np.min(avg_luminosity))
            }
        }
        
        logger.info(f"Luminosity analysis complete. {len(anomalies)} anomalies detected.")
        return results
    
    def _analyze_pulse_patterns(self, luminosity_data, fps):
        """Analyze potential rhythmic pulse patterns."""
        if len(luminosity_data) < fps:  # Need at least 1 second of data
            return {'detected': False, 'reason': 'insufficient_data'}
        
        # Normalize the data
        normalized = np.array(luminosity_data)
        normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
        
        # Apply bandpass filter to isolate potential pulses
        pulse_config = self.config.get('pulse_analysis', {})
        freq_range = pulse_config.get('filter_frequency_range', [0.5, 10.0])
        
        nyquist = fps / 2
        low = freq_range[0] / nyquist
        high = freq_range[1] / nyquist
        
        if low >= 1.0 or high >= 1.0:
            return {'detected': False, 'reason': 'invalid_frequency_range'}
        
        # Design and apply bandpass filter
        filter_order = pulse_config.get('filter_order', 4)
        b, a = butter(filter_order, [low, high], btype='band')
        filtered = filtfilt(b, a, normalized)
        
        # Detect peaks in filtered signal
        peak_height = pulse_config.get('peak_detection_height', 0.2)
        min_distance = pulse_config.get('peak_min_distance_frames', fps // 8)
        
        peaks, properties = find_peaks(
            filtered, 
            height=peak_height,
            distance=min_distance
        )
        
        if len(peaks) < 3:  # Need at least 3 peaks for pattern analysis
            return {'detected': False, 'reason': 'insufficient_peaks'}
        
        # Calculate timing statistics
        peak_times = peaks / fps
        intervals = np.diff(peak_times)
        avg_interval = np.mean(intervals)
        frequency = 1 / avg_interval if avg_interval > 0 else 0
        
        # Calculate regularity (coefficient of variation)
        interval_cv = np.std(intervals) / avg_interval if avg_interval > 0 else float('inf')
        
        return {
            'detected': True,
            'peak_count': len(peaks),
            'peak_indices': peaks.tolist(),
            'peak_times': peak_times.tolist(),
            'intervals': intervals.tolist(),
            'average_interval': float(avg_interval),
            'frequency_hz': float(frequency),
            'regularity_score': float(1 / (1 + interval_cv)),  # Higher = more regular
            'filtered_signal': filtered.tolist()
        }
    
    def _detect_luminosity_anomalies(self, luminosity_data, timestamps):
        """Detect unusual luminosity events."""
        if len(luminosity_data) < 10:
            return []
        
        # Calculate statistical thresholds
        mean_lum = np.mean(luminosity_data)
        std_lum = np.std(luminosity_data)
        
        # Define thresholds for anomalies
        upper_threshold = mean_lum + 3 * std_lum
        lower_threshold = mean_lum - 3 * std_lum
        
        anomalies = []
        for i, (lum, timestamp) in enumerate(zip(luminosity_data, timestamps)):
            if lum > upper_threshold:
                anomalies.append({
                    'type': 'bright_flash',
                    'timestamp': timestamp,
                    'frame': i,
                    'value': float(lum),
                    'severity': float((lum - mean_lum) / std_lum)
                })
            elif lum < lower_threshold:
                anomalies.append({
                    'type': 'darkness_anomaly',
                    'timestamp': timestamp,
                    'frame': i,
                    'value': float(lum),
                    'severity': float((mean_lum - lum) / std_lum)
                })
        
        return anomalies
    
    def create_luminosity_visualization(self, analysis_results, output_path):
        """Create visualization plots for luminosity analysis."""
        import matplotlib.pyplot as plt
        
        timestamps = analysis_results['timestamps']
        avg_lum = analysis_results['average_luminosity']
        center_lum = analysis_results['center_luminosity']
        pulse_data = analysis_results['pulse_analysis']
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Overall luminosity
        axes[0].plot(timestamps, avg_lum, label='Average', linewidth=1)
        axes[0].plot(timestamps, center_lum, label='Center Region', linewidth=1)
        axes[0].set_title('Luminosity Over Time')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Luminosity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Min/Max range
        axes[1].plot(timestamps, analysis_results['maximum_luminosity'], label='Maximum', linewidth=1)
        axes[1].plot(timestamps, analysis_results['minimum_luminosity'], label='Minimum', linewidth=1)
        axes[1].fill_between(timestamps, 
                           analysis_results['minimum_luminosity'],
                           analysis_results['maximum_luminosity'],
                           alpha=0.3, label='Range')
        axes[1].set_title('Luminosity Range')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Luminosity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Pulse analysis (if detected)
        if pulse_data.get('detected'):
            filtered_signal = pulse_data['filtered_signal']
            peak_times = pulse_data['peak_times']
            peak_indices = pulse_data['peak_indices']
            
            axes[2].plot(timestamps, filtered_signal, label='Filtered Signal', linewidth=1)
            axes[2].scatter([timestamps[i] for i in peak_indices], 
                          [filtered_signal[i] for i in peak_indices],
                          color='red', s=30, label='Detected Peaks', zorder=5)
            axes[2].set_title(f'Pulse Pattern Analysis (Freq: {pulse_data["frequency_hz"]:.2f} Hz)')
            axes[2].set_xlabel('Time (seconds)')
            axes[2].set_ylabel('Filtered Intensity')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No significant pulse pattern detected', 
                        transform=axes[2].transAxes, ha='center', va='center',
                        fontsize=14)
            axes[2].set_title('Pulse Pattern Analysis')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Luminosity visualization saved to {output_path}")
        return str(output_path)