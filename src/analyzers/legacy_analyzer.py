#!/usr/bin/env python3
"""
UAP Analyzer Tool - Comprehensive UAP/Unusual Aerial Phenomena Video Analysis Tool

This tool performs multiple forms of analysis on video footage:
1. Frame extraction and basic metadata extraction
2. Motion tracking and trajectory analysis
3. Light/luminosity pattern analysis
4. 3D depth estimation and spatial analysis
5. EM noise simulation/visualization
6. Spectral analysis
7. Statistical anomaly detection

Usage: python uap_analyzer_tool.py <input_video> [options]
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import json
import argparse
import datetime
from tqdm import tqdm

class UAPAnalyzer:
    def __init__(self, input_video, output_dir=None, verbose=False):
        """Initialize the UAP analyzer with input video path."""
        self.input_path = input_video
        
        # Create output directory based on input filename if not specified
        if output_dir is None:
            base_name = os.path.splitext(os.path.basename(input_video))[0]
            self.output_dir = f"uap_analysis_{base_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.output_dir = output_dir
            
        # Create directory structure
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.enhanced_dir = os.path.join(self.output_dir, "enhanced")
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.input_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps
        
        # Analysis state
        self.frames = []
        self.timestamps = []
        self.luminosity_data = []
        self.motion_data = []
        self.depth_maps = []
        self.verbose = verbose
        
        # Create directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.enhanced_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        self.print_verbose(f"Initialized UAP Analyzer for {input_video}")
        self.print_verbose(f"Video properties: {self.width}x{self.height}, {self.fps} fps, {self.duration:.2f} seconds, {self.frame_count} frames")
    
    def print_verbose(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def extract_frames(self, extract_all=False, max_frames=None):
        """Extract frames from video with optional sampling."""
        self.print_verbose("Extracting frames...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if max_frames is not None and max_frames < self.frame_count:
            step = max(1, self.frame_count // max_frames)
            total_frames = min(max_frames, self.frame_count)
        else:
            step = 1
            total_frames = self.frame_count
        
        extracted_frames = 0
        frames_extracted = []
        
        progress_bar = tqdm(total=total_frames, desc="Extracting frames", 
                           disable=not self.verbose)
        
        while extracted_frames < total_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp = current_frame / self.fps
            
            if current_frame % step == 0:
                # Store frame and timestamp
                frames_extracted.append(frame)
                self.timestamps.append(timestamp)
                
                # Save frame to disk if extract_all is True
                if extract_all:
                    frame_filename = os.path.join(self.frames_dir, f"frame_{extracted_frames:08d}.png")
                    cv2.imwrite(frame_filename, frame)
                
                extracted_frames += 1
                progress_bar.update(1)
                
                # Skip frames according to step size
                if step > 1 and extracted_frames < total_frames:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + step)
        
        progress_bar.close()
        self.frames = frames_extracted
        self.print_verbose(f"Extracted {len(self.frames)} frames")
        return self.frames
    
    def analyze_motion(self):
        """Analyze motion between consecutive frames."""
        self.print_verbose("Analyzing motion...")
        if not self.frames:
            self.print_verbose("No frames extracted. Running frame extraction...")
            self.extract_frames()
        
        # Initialize motion tracking
        motion_vectors = []
        motion_energy = []
        frame_diff = []
        
        prev_gray = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(self.frames[0])
        hsv[..., 1] = 255  # Set saturation to maximum
        
        progress_bar = tqdm(total=len(self.frames)-1, desc="Analyzing motion", 
                           disable=not self.verbose)
        
        for i in range(1, len(self.frames)):
            # Convert frame to grayscale
            gray = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Calculate magnitude and direction
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_energy.append(np.mean(mag))
            
            # Calculate frame difference
            diff = cv2.absdiff(prev_gray, gray)
            frame_diff.append(np.mean(diff))
            
            # Store motion data
            motion_vectors.append(flow)
            
            # Update previous frame
            prev_gray = gray
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Store results
        self.motion_data = {
            "vectors": motion_vectors,
            "energy": motion_energy,
            "frame_diff": frame_diff
        }
        
        # Save motion energy graph
        plt.figure(figsize=(15, 5))
        plt.plot(self.timestamps[1:], motion_energy)
        plt.title("Motion Energy Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Average Motion Magnitude")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.analysis_dir, "motion_energy.png"), dpi=150)
        plt.close()
        
        # Save frame difference graph
        plt.figure(figsize=(15, 5))
        plt.plot(self.timestamps[1:], frame_diff)
        plt.title("Frame Difference Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Average Pixel Difference")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.analysis_dir, "frame_difference.png"), dpi=150)
        plt.close()
        
        # Save motion data
        np.save(os.path.join(self.analysis_dir, "motion_data.npy"), {
            "timestamps": self.timestamps[1:],
            "energy": motion_energy,
            "frame_diff": frame_diff
        })
        
        self.print_verbose(f"Motion analysis complete. Data saved to {self.analysis_dir}")
        return self.motion_data
    
    def analyze_luminosity(self):
        """Analyze light patterns and luminosity changes."""
        self.print_verbose("Analyzing luminosity patterns...")
        if not self.frames:
            self.print_verbose("No frames extracted. Running frame extraction...")
            self.extract_frames()
        
        # Initialize luminosity metrics
        avg_luminosity = []
        max_luminosity = []
        min_luminosity = []
        center_luminosity = []
        
        # Define center region (25% of frame)
        center_h_start = self.height // 3
        center_h_end = (self.height // 3) * 2
        center_w_start = self.width // 3
        center_w_end = (self.width // 3) * 2
        
        progress_bar = tqdm(total=len(self.frames), desc="Analyzing luminosity", 
                           disable=not self.verbose)
        
        for frame in self.frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate luminosity metrics
            avg_luminosity.append(np.mean(gray))
            max_luminosity.append(np.max(gray))
            min_luminosity.append(np.min(gray))
            
            # Calculate center region luminosity
            center = gray[center_h_start:center_h_end, center_w_start:center_w_end]
            center_luminosity.append(np.mean(center))
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Store results
        self.luminosity_data = {
            "timestamps": self.timestamps,
            "average": avg_luminosity,
            "maximum": max_luminosity,
            "minimum": min_luminosity,
            "center": center_luminosity
        }
        
        # Save to JSON
        with open(os.path.join(self.analysis_dir, "luminosity_data.json"), 'w') as f:
            json.dump({
                "timestamps": self.timestamps,
                "average": avg_luminosity,
                "maximum": max_luminosity,
                "minimum": min_luminosity,
                "center": center_luminosity
            }, f, indent=2)
        
        # Generate luminosity graph
        plt.figure(figsize=(15, 10))
        
        # Plot average luminosity
        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, avg_luminosity, label="Average")
        plt.plot(self.timestamps, center_luminosity, label="Center Region")
        plt.title("Luminosity Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Average Luminosity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot min/max luminosity
        plt.subplot(2, 1, 2)
        plt.plot(self.timestamps, max_luminosity, label="Maximum")
        plt.plot(self.timestamps, min_luminosity, label="Minimum")
        plt.title("Min/Max Luminosity Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Luminosity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, "luminosity_analysis.png"), dpi=150)
        plt.close()
        
        # Analyze light pulse patterns
        self.analyze_light_pulses(center_luminosity)
        
        self.print_verbose(f"Luminosity analysis complete. Data saved to {self.analysis_dir}")
        return self.luminosity_data
    
    def analyze_light_pulses(self, luminosity_data):
        """Analyze potential light pulse patterns in the luminosity data."""
        self.print_verbose("Analyzing light pulse patterns...")
        
        # Normalize the data
        normalized = np.array(luminosity_data)
        normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
        
        # Apply a bandpass filter to isolate potential pulses (0.5-10 Hz)
        nyquist = self.fps / 2
        low = 0.5 / nyquist
        high = 10.0 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, normalized)
        
        # Find peaks in the filtered data
        peaks, _ = find_peaks(filtered, height=0.2, distance=self.fps/8)
        
        # Calculate time between peaks (if any found)
        if len(peaks) > 1:
            peak_times = [self.timestamps[p] for p in peaks]
            intervals = np.diff(peak_times)
            avg_interval = np.mean(intervals)
            frequency = 1 / avg_interval if avg_interval > 0 else 0
            
            pulse_data = {
                "peak_indices": peaks.tolist(),
                "peak_times": peak_times,
                "intervals": intervals.tolist(),
                "average_interval": float(avg_interval),
                "frequency_hz": float(frequency)
            }
            
            # Save pulse data
            with open(os.path.join(self.analysis_dir, "pulse_data.json"), 'w') as f:
                json.dump(pulse_data, f, indent=2)
            
            # Plot pulse analysis
            plt.figure(figsize=(15, 10))
            
            # Original luminosity
            plt.subplot(3, 1, 1)
            plt.plot(self.timestamps, normalized)
            plt.title("Normalized Luminosity")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Normalized Value")
            plt.grid(True, alpha=0.3)
            
            # Filtered signal
            plt.subplot(3, 1, 2)
            plt.plot(self.timestamps, filtered)
            plt.plot(self.timestamps[peaks], filtered[peaks], "x", color='red')
            plt.title(f"Filtered Signal with Detected Pulses (Avg Freq: {frequency:.2f} Hz)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Filtered Value")
            plt.grid(True, alpha=0.3)
            
            # Interval histogram
            plt.subplot(3, 1, 3)
            plt.hist(intervals, bins=20)
            plt.axvline(x=avg_interval, color='r', linestyle='--', 
                      label=f'Avg: {avg_interval:.3f}s ({frequency:.2f} Hz)')
            plt.title("Distribution of Pulse Intervals")
            plt.xlabel("Interval (seconds)")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.analysis_dir, "pulse_pattern_analysis.png"), dpi=150)
            plt.close()
            
            # Create a custom visualization of the pulse pattern
            self.visualize_pulse_pattern(normalized, filtered, peaks)
            
            self.print_verbose(f"Detected light pulse frequency: {frequency:.2f} Hz")
        else:
            self.print_verbose("No significant light pulse patterns detected")
    
    def visualize_pulse_pattern(self, normalized, filtered, peaks):
        """Create a visualization of the detected pulse pattern."""
        # Create time array for full duration
        t = np.linspace(0, self.duration, 1000)
        
        # If we have peaks, try to model the pattern
        if len(peaks) > 1:
            # Calculate average interval and frequency
            peak_times = [self.timestamps[p] for p in peaks]
            intervals = np.diff(peak_times)
            avg_interval = np.mean(intervals)
            frequency = 1 / avg_interval if avg_interval > 0 else 0
            
            # Generate idealized pattern
            modulation_freq = frequency / 4  # Estimate a slower modulation
            base_pattern = 0.5 + 0.3 * np.sin(2 * np.pi * frequency * t)
            modulation = 0.2 * np.sin(2 * np.pi * modulation_freq * t)
            idealized = base_pattern + modulation
            
            # Create visualization
            plt.figure(figsize=(15, 6))
            plt.plot(t, idealized, "g-", linewidth=1.5, label="Idealized Pattern")
            plt.plot(self.timestamps, normalized, "b-", linewidth=0.7, alpha=0.5, label="Actual Luminosity")
            plt.plot(self.timestamps[peaks], normalized[peaks], "ro", markersize=4, label="Detected Peaks")
            
            plt.grid(True, alpha=0.3)
            plt.title(f"UAP Light Pulse Pattern (Est. Frequency: {frequency:.2f} Hz)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Normalized Light Intensity")
            plt.legend()
            
            # Add markers for key timestamp ranges (first third, middle third, last third)
            third_duration = self.duration / 3
            plt.axvspan(0, third_duration, color='blue', alpha=0.05)
            plt.axvspan(third_duration, 2*third_duration, color='purple', alpha=0.05)
            plt.axvspan(2*third_duration, self.duration, color='red', alpha=0.05)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.analysis_dir, "pulse_pattern.png"), dpi=300)
            plt.close()
    
    def generate_spectral_analysis(self):
        """Generate spectral analysis of the video."""
        self.print_verbose("Generating spectral analysis...")
        if not self.frames:
            self.print_verbose("No frames extracted. Running frame extraction...")
            self.extract_frames()
        
        # Create a spectral waterfall plot
        frame_count = len(self.frames)
        
        # We'll analyze the average spectral characteristics of each frame
        r_spectrum = np.zeros((frame_count, 256))
        g_spectrum = np.zeros((frame_count, 256))
        b_spectrum = np.zeros((frame_count, 256))
        
        progress_bar = tqdm(total=frame_count, desc="Spectral analysis", 
                           disable=not self.verbose)
        
        for i, frame in enumerate(self.frames):
            # Split into color channels
            b, g, r = cv2.split(frame)
            
            # Calculate histograms
            r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
            g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
            b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
            
            # Normalize and store
            r_spectrum[i] = r_hist.flatten() / r_hist.max()
            g_spectrum[i] = g_hist.flatten() / g_hist.max()
            b_spectrum[i] = b_hist.flatten() / b_hist.max()
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Save the spectral data
        np.save(os.path.join(self.analysis_dir, "spectral_data.npy"), {
            "timestamps": self.timestamps,
            "r_spectrum": r_spectrum,
            "g_spectrum": g_spectrum,
            "b_spectrum": b_spectrum
        })
        
        # Create spectral waterfall visualization
        plt.figure(figsize=(15, 10))
        
        # Red channel
        plt.subplot(3, 1, 1)
        plt.imshow(r_spectrum.T, aspect='auto', origin='lower', 
                  extent=[0, self.duration, 0, 255],
                  cmap='inferno')
        plt.colorbar(label='Normalized Intensity')
        plt.title("Red Channel Spectral Evolution")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Intensity Level")
        
        # Green channel
        plt.subplot(3, 1, 2)
        plt.imshow(g_spectrum.T, aspect='auto', origin='lower', 
                  extent=[0, self.duration, 0, 255],
                  cmap='inferno')
        plt.colorbar(label='Normalized Intensity')
        plt.title("Green Channel Spectral Evolution")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Intensity Level")
        
        # Blue channel
        plt.subplot(3, 1, 3)
        plt.imshow(b_spectrum.T, aspect='auto', origin='lower', 
                  extent=[0, self.duration, 0, 255],
                  cmap='inferno')
        plt.colorbar(label='Normalized Intensity')
        plt.title("Blue Channel Spectral Evolution")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Intensity Level")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, "spectral_analysis.png"), dpi=150)
        plt.close()
        
        self.print_verbose(f"Spectral analysis complete. Data saved to {self.analysis_dir}")
    
    def simulate_em_noise(self):
        """Simulate and visualize potential EM effects."""
        self.print_verbose("Simulating EM noise visualization...")
        if not self.frames:
            self.print_verbose("No frames extracted. Running frame extraction...")
            self.extract_frames()
        
        # Create simulated EM noise videos
        output_path = os.path.join(self.analysis_dir, "em_noise_simulation.mp4")
        
        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.width, self.height))
        
        # Create custom colormap for EM visualization
        colors = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.7), 
                 (0.0, 0.7, 0.7), (0.7, 0.7, 0.0),
                 (0.7, 0.0, 0.0), (1.0, 1.0, 1.0)]
        cm_name = 'em_noise'
        em_cmap = LinearSegmentedColormap.from_list(cm_name, colors, N=256)
        
        progress_bar = tqdm(total=len(self.frames), desc="Simulating EM noise", 
                           disable=not self.verbose)
        
        for i, frame in enumerate(self.frames):
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
            
            # Apply edge detection to identify object boundaries
            edges = cv2.Canny(frame, 50, 150)
            
            # Create gaussian noise that's stronger near edges
            noise_mask = gaussian_filter(edges.astype(float), sigma=15) / 255.0
            noise = np.random.normal(0, 20, gray.shape) * noise_mask
            
            # Add time-dependent wave pattern
            x = np.linspace(0, self.width-1, self.width)
            y = np.linspace(0, self.height-1, self.height)
            X, Y = np.meshgrid(x, y)
            
            # Create wave patterns
            t = self.timestamps[i]
            wave1 = np.sin(0.1 * X + 0.2 * Y + 2 * t)
            wave2 = np.sin(0.05 * X - 0.1 * Y + t)
            waves = (wave1 + wave2) / 2
            
            # Combine with noise
            combined = noise + 20 * waves * noise_mask
            
            # Normalize and apply to original frame
            norm_combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
            
            # Generate colormap image
            cm_img = plt.cm.get_cmap(em_cmap)(norm_combined.astype(int))
            cm_img = (cm_img[:, :, :3] * 255).astype(np.uint8)
            
            # Blend with original image
            alpha = 0.7
            blend = cv2.addWeighted(frame, 1-alpha, cm_img, alpha, 0)
            
            # Write to video
            out.write(blend)
            progress_bar.update(1)
        
        progress_bar.close()
        out.release()
        
        self.print_verbose(f"EM noise simulation complete. Video saved to {output_path}")
    
    def generate_motion_tracking_video(self):
        """Generate a video with motion tracking visualization."""
        self.print_verbose("Generating motion tracking visualization...")
        
        if not self.frames:
            self.print_verbose("No frames extracted. Running frame extraction...")
            self.extract_frames()
            
        if not hasattr(self, 'motion_data') or not self.motion_data:
            self.print_verbose("No motion data. Running motion analysis...")
            self.analyze_motion()
        
        # Define output path
        output_path = os.path.join(self.analysis_dir, "motion_tracking.mp4")
        
        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.width, self.height))
        
        # Get motion vectors
        motion_vectors = self.motion_data["vectors"]
        
        # Parameters for visualization
        step = 16  # Grid step
        
        # Prepare first frame
        prev_frame = self.frames[0].copy()
        out.write(prev_frame)
        
        progress_bar = tqdm(total=len(motion_vectors), desc="Creating motion tracking video", 
                           disable=not self.verbose)
        
        # Create grid for visualization
        h, w = self.frames[0].shape[:2]
        y_grid, x_grid = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        
        for i, flow in enumerate(motion_vectors):
            # Get current frame
            frame = self.frames[i+1].copy()
            
            # Display frame number and timestamp
            cv2.putText(frame, f"Frame: {i+1}/{len(self.frames)-1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {self.timestamps[i+1]:.2f}s", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Extract flow vectors
            fx, fy = flow[y_grid, x_grid].T
            
            # Create visualization
            lines = np.vstack([x_grid, y_grid, (x_grid + fx).astype(int), (y_grid + fy).astype(int)]).T.reshape(-1, 2, 2)
            
            # Calculate speeds for color coding
            speeds = np.sqrt(fx*fx + fy*fy)
            max_speed = max(1, speeds.max())
            
            # Draw the flow vectors
            for (x1, y1), (x2, y2), speed in zip(lines[:, 0], lines[:, 1], speeds):
                # Skip small motions
                if speed < 0.5:
                    continue
                
                # Color based on speed (red=fast, blue=slow)
                color = (int(255 * (1 - speed / max_speed)), 0, int(255 * speed / max_speed))
                
                # Draw line and circle
                cv2.arrowedLine(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, tipLength=0.3)
            
            # Write frame to video
            out.write(frame)
            progress_bar.update(1)
        
        progress_bar.close()
        out.release()
        
        self.print_verbose(f"Motion tracking video saved to {output_path}")
    
    def generate_enhanced_video(self):
        """Generate an enhanced version of the video with better visibility."""
        self.print_verbose("Generating enhanced video...")
        
        if not self.frames:
            self.print_verbose("No frames extracted. Running frame extraction...")
            self.extract_frames()
        
        # Define output paths
        output_path = os.path.join(self.enhanced_dir, "enhanced_video.mp4")
        
        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.width, self.height))
        
        progress_bar = tqdm(total=len(self.frames), desc="Creating enhanced video", 
                           disable=not self.verbose)
        
        for frame in self.frames:
            # Create a copy for enhancement
            enhanced = frame.copy()
            
            # Convert to LAB space
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to luminance channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab_clahe = cv2.merge((l_clahe, a, b))
            enhanced_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            
            # Apply sharpening
            kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
            enhanced_sharp = cv2.filter2D(enhanced_clahe, -1, kernel)
            
            # Apply denoising
            enhanced_denoised = cv2.fastNlMeansDenoisingColored(enhanced_sharp, None, 5, 5, 7, 21)
            
            # Write to video
            out.write(enhanced_denoised)
            progress_bar.update(1)
        
        progress_bar.close()
        out.release()
        
        self.print_verbose(f"Enhanced video saved to {output_path}")
    
    def create_stabilized_video(self):
        """Create a stabilized version of the video."""
        self.print_verbose("Creating stabilized video...")
        
        if not self.frames:
            self.print_verbose("No frames extracted. Running frame extraction...")
            self.extract_frames()
        
        # Define output path
        output_path = os.path.join(self.enhanced_dir, "stabilized_video.mp4")
        
        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.width, self.height))
        
        # Convert first frame to grayscale
        prev_gray = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Initialize parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15),
                       maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Create matrix for transformation
        prev_pts = None
        transform_matrices = []
        
        # First frame is reference
        transform_matrices.append(np.eye(2, 3, dtype=np.float32))
        out.write(self.frames[0])
        
        progress_bar = tqdm(total=len(self.frames)-1, desc="Stabilizing video", 
                           disable=not self.verbose)
        
        for i in range(1, len(self.frames)):
            # Convert to grayscale
            gray = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
            
            # If we don't have previous points, detect features
            if prev_pts is None:
                prev_kp = sift.detect(prev_gray, None)
                prev_pts = np.array([kp.pt for kp in prev_kp], dtype=np.float32).reshape(-1, 1, 2)
            
            # Calculate optical flow to track points
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            
            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_good_pts = prev_pts[idx].reshape(-1, 2)
            curr_good_pts = curr_pts[idx].reshape(-1, 2)
            
            # Make sure we have enough points
            if len(prev_good_pts) < 4 or len(curr_good_pts) < 4:
                # Not enough points, use identity transformation
                transform_matrices.append(np.eye(2, 3, dtype=np.float32))
            else:
                # Find transformation matrix
                transform = cv2.estimateAffinePartial2D(prev_good_pts, curr_good_pts)[0]
                
                # If transformation is None, use identity
                if transform is None:
                    transform = np.eye(2, 3, dtype=np.float32)
                
                transform_matrices.append(transform)
            
            # Update for next iteration
            prev_gray = gray
            prev_pts = curr_pts
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Apply smoothing to transformations
        smoothed_transforms = []
        window_size = 15
        
        for i in range(len(transform_matrices)):
            start = max(0, i - window_size // 2)
            end = min(len(transform_matrices), i + window_size // 2 + 1)
            
            # Compute weighted average of nearby transforms
            weights = np.exp(-0.5 * ((np.arange(start, end) - i) / (window_size / 4)) ** 2)
            avg_transform = np.zeros((2, 3), dtype=np.float32)
            
            for j, w in enumerate(weights):
                avg_transform += transform_matrices[start + j] * w
            
            avg_transform /= weights.sum()
            smoothed_transforms.append(avg_transform)
        
        # Apply stabilization and write video
        progress_bar = tqdm(total=len(self.frames), desc="Writing stabilized video", 
                           disable=not self.verbose)
        
        for i, frame in enumerate(self.frames):
            # Apply stabilization transformation
            stabilized = cv2.warpAffine(frame, smoothed_transforms[i], (self.width, self.height))
            
            # Write to video
            out.write(stabilized)
            progress_bar.update(1)
        
        progress_bar.close()
        out.release()
        
        self.print_verbose(f"Stabilized video saved to {output_path}")
    
    def create_summary_report(self):
        """Create a summary report of all analysis."""
        self.print_verbose("Creating summary report...")
        
        # Gather summary data
        summary = {
            "video_properties": {
                "filename": os.path.basename(self.input_path),
                "resolution": f"{self.width}x{self.height}",
                "fps": self.fps,
                "frame_count": self.frame_count,
                "duration": self.duration
            },
            "analysis_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add luminosity data if available
        if hasattr(self, 'luminosity_data') and self.luminosity_data:
            # Calculate average luminosity
            avg_lum = np.mean(self.luminosity_data["average"])
            
            # Check for peaks in center luminosity if available
            if "center" in self.luminosity_data:
                center_lum = np.array(self.luminosity_data["center"])
                center_lum_norm = (center_lum - np.min(center_lum)) / (np.max(center_lum) - np.min(center_lum))
                
                # Apply bandpass filter
                nyquist = self.fps / 2
                low = 0.5 / nyquist
                high = 10.0 / nyquist
                b, a = butter(4, [low, high], btype='band')
                filtered = filtfilt(b, a, center_lum_norm)
                
                # Find peaks
                peaks, _ = find_peaks(filtered, height=0.2, distance=self.fps/8)
                
                if len(peaks) > 1:
                    peak_times = [self.timestamps[p] for p in peaks]
                    intervals = np.diff(peak_times)
                    avg_interval = np.mean(intervals)
                    frequency = 1 / avg_interval if avg_interval > 0 else 0
                    
                    summary["light_pulses"] = {
                        "detected": True,
                        "peak_count": len(peaks),
                        "frequency_hz": float(frequency),
                        "average_interval": float(avg_interval)
                    }
                else:
                    summary["light_pulses"] = {
                        "detected": False
                    }
        
        # Add motion data if available
        if hasattr(self, 'motion_data') and self.motion_data:
            if "energy" in self.motion_data:
                motion_energy = self.motion_data["energy"]
                
                # Find significant motion events
                if len(motion_energy) > 0:
                    mean_energy = np.mean(motion_energy)
                    std_energy = np.std(motion_energy)
                    threshold = mean_energy + 2 * std_energy
                    
                    # Find frames with significant motion
                    significant_frames = np.where(np.array(motion_energy) > threshold)[0]
                    
                    if len(significant_frames) > 0:
                        # Convert to timestamps
                        significant_times = [self.timestamps[i+1] for i in significant_frames]
                        
                        summary["motion_events"] = {
                            "significant_motion_count": len(significant_frames),
                            "timestamps": significant_times,
                            "average_motion": float(mean_energy)
                        }
        
        # Write summary to JSON
        summary_path = os.path.join(self.analysis_dir, "analysis_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create a human-readable text report
        report_path = os.path.join(self.output_dir, "analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("UAP VIDEO ANALYSIS REPORT\n")
            f.write("=========================\n\n")
            f.write(f"Analysis Date: {summary['analysis_timestamp']}\n\n")
            
            f.write("VIDEO PROPERTIES\n")
            f.write("----------------\n")
            f.write(f"Filename: {summary['video_properties']['filename']}\n")
            f.write(f"Resolution: {summary['video_properties']['resolution']}\n")
            f.write(f"Frame Rate: {summary['video_properties']['fps']} fps\n")
            f.write(f"Frame Count: {summary['video_properties']['frame_count']}\n")
            f.write(f"Duration: {summary['video_properties']['duration']:.2f} seconds\n\n")
            
            if "light_pulses" in summary:
                f.write("LIGHT PATTERN ANALYSIS\n")
                f.write("---------------------\n")
                if summary["light_pulses"]["detected"]:
                    f.write(f"Light Pulse Pattern Detected: Yes\n")
                    f.write(f"Number of Peaks: {summary['light_pulses']['peak_count']}\n")
                    f.write(f"Estimated Frequency: {summary['light_pulses']['frequency_hz']:.2f} Hz\n")
                    f.write(f"Average Interval: {summary['light_pulses']['average_interval']:.3f} seconds\n\n")
                else:
                    f.write("Light Pulse Pattern Detected: No\n\n")
            
            if "motion_events" in summary:
                f.write("MOTION ANALYSIS\n")
                f.write("---------------\n")
                f.write(f"Significant Motion Events: {summary['motion_events']['significant_motion_count']}\n")
                f.write("Timestamps of Significant Motion:\n")
                for t in summary["motion_events"]["timestamps"]:
                    f.write(f"  - {t:.2f}s\n")
                f.write(f"Average Motion Energy: {summary['motion_events']['average_motion']:.4f}\n\n")
            
            f.write("FILES GENERATED\n")
            f.write("--------------\n")
            
            # List generated files by directory
            for root, dirs, files in os.walk(self.output_dir):
                rel_path = os.path.relpath(root, self.output_dir)
                if rel_path == ".":
                    f.write("Main Directory:\n")
                else:
                    f.write(f"{rel_path}:\n")
                
                for file in sorted(files):
                    if file != "analysis_report.txt":  # Skip this file
                        f.write(f"  - {file}\n")
                f.write("\n")
        
        self.print_verbose(f"Summary report saved to {report_path}")
    
    def run_all_analyses(self):
        """Run all analysis methods."""
        self.print_verbose("Running all analyses...")
        
        # Extract frames
        self.extract_frames(extract_all=True)
        
        # Run analyses
        self.analyze_motion()
        self.analyze_luminosity()
        self.generate_spectral_analysis()
        
        # Generate visual outputs
        self.generate_motion_tracking_video()
        self.generate_enhanced_video()
        self.create_stabilized_video()
        self.simulate_em_noise()
        
        # Create summary
        self.create_summary_report()
        
        self.print_verbose(f"All analyses complete. Results saved to {self.output_dir}")
        
        # Print a summary for terminal output
        print("\n" + "="*80)
        print(f"UAP Analysis Complete for {os.path.basename(self.input_path)}")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print("\nGenerated files:")
        print(f"  - Analysis data: {self.analysis_dir}")
        print(f"  - Enhanced videos: {self.enhanced_dir}")
        print(f"  - Extracted frames: {self.frames_dir}")
        print(f"  - Summary report: {os.path.join(self.output_dir, 'analysis_report.txt')}")
        print("\nTo view the full analysis report:")
        print(f"  cat {os.path.join(self.output_dir, 'analysis_report.txt')}")
        print("="*80)

def display_ascii_header():
    """Display an ASCII art header for the tool."""
    print("""
 █    ██  ▄▄▄       ██▓███      ▄▄▄       ███▄    █  ▄▄▄       ██▓   ▓██   ██▓▒███████▒▓█████  ██▀███  
 ██  ▓██▒▒████▄    ▓██░  ██▒   ▒████▄     ██ ▀█   █ ▒████▄    ▓██▒    ▒██  ██▒▒ ▒ ▒ ▄▀░▓█   ▀ ▓██ ▒ ██▒
▓██  ▒██░▒██  ▀█▄  ▓██░ ██▓▒   ▒██  ▀█▄  ▓██  ▀█ ██▒▒██  ▀█▄  ▒██░     ▒██ ██░░ ▒ ▄▀▒░ ▒███   ▓██ ░▄█ ▒
▓▓█  ░██░░██▄▄▄▄██ ▒██▄█▓▒ ▒   ░██▄▄▄▄██ ▓██▒  ▐▌██▒░██▄▄▄▄██ ▒██░     ░ ▐██▓░  ▄▀▒   ░▒▓█  ▄ ▒██▀▀█▄  
▒▒█████▓  ▓█   ▓██▒▒██▒ ░  ░    ▓█   ▓██▒▒██░   ▓██░ ▓█   ▓██▒░██████▒ ░ ██▒▓░▒███████▒░▒████▒░██▓ ▒██▒
░▒▓▒ ▒ ▒  ▒▒   ▓▒█░▒▓▒░ ░  ░    ▒▒   ▓▒█░░ ▒░   ▒ ▒  ▒▒   ▓▒█░░ ▒░▓  ░  ██▒▒▒ ░▒▒ ▓░▒░▒░░ ▒░ ░░ ▒▓ ░▒▓░
░░▒░ ░ ░   ▒   ▒▒ ░░▒ ░          ▒   ▒▒ ░░ ░░   ░ ▒░  ▒   ▒▒ ░░ ░ ▒  ░▓██ ░▒░ ░░▒ ▒ ░ ▒ ░ ░  ░  ░▒ ░ ▒░
 ░░░ ░ ░   ░   ▒   ░░            ░   ▒      ░   ░ ░   ░   ▒     ░ ░   ▒ ▒ ░░  ░ ░ ░ ░ ░   ░     ░░   ░ 
   ░           ░  ░                  ░  ░         ░       ░  ░    ░  ░░ ░     ░ ░       ░  ░   ░      
                                                                      ░ ░     ░                        
    """)
    print("UAP Video Analysis Tool - v1.0")
    print("=" * 80)

def main():
    """Main function to parse command line arguments and run analysis."""
    parser = argparse.ArgumentParser(description='UAP Video Analysis Tool')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('-o', '--output', help='Output directory for analysis results')
    parser.add_argument('-f', '--frames', type=int, help='Maximum number of frames to extract (default: all)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    # Analysis options
    parser.add_argument('--no-enhanced', action='store_true', help='Skip enhanced video generation')
    parser.add_argument('--no-stabilized', action='store_true', help='Skip stabilized video generation')
    parser.add_argument('--no-em', action='store_true', help='Skip EM noise simulation')
    parser.add_argument('--no-tracking', action='store_true', help='Skip motion tracking video')
    
    args = parser.parse_args()
    
    # Display ASCII header
    display_ascii_header()
    
    # Check if input file exists
    if not os.path.isfile(args.input_video):
        print(f"Error: Input video file '{args.input_video}' not found.")
        return 1
    
    try:
        # Initialize analyzer
        analyzer = UAPAnalyzer(args.input_video, args.output, args.verbose)
        
        # Extract frames
        analyzer.extract_frames(extract_all=True, max_frames=args.frames)
        
        # Run analyses
        analyzer.analyze_motion()
        analyzer.analyze_luminosity()
        analyzer.generate_spectral_analysis()
        
        # Generate visual outputs
        if not args.no_tracking:
            analyzer.generate_motion_tracking_video()
        
        if not args.no_enhanced:
            analyzer.generate_enhanced_video()
        
        if not args.no_stabilized:
            analyzer.create_stabilized_video()
        
        if not args.no_em:
            analyzer.simulate_em_noise()
        
        # Create summary
        analyzer.create_summary_report()
        
        print(f"\nAnalysis complete. Results saved to {analyzer.output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())