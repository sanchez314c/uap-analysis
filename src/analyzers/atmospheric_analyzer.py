#!/usr/bin/env python3
"""
Atmospheric Analysis Component
=============================

Analyzes atmospheric disturbances, air displacement, and environmental interactions
that may indicate object presence or propulsion effects.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

class AtmosphericAnalyzer:
    """Analyzes atmospheric disturbances and air displacement patterns."""
    
    def __init__(self, config):
        """Initialize atmospheric analyzer."""
        self.config = config
        
    def analyze(self, frames, metadata):
        """Analyze atmospheric effects in video frames."""
        logger.info("Starting atmospheric analysis...")
        
        results = {
            'heat_distortion': self._analyze_heat_distortion(frames),
            'air_displacement': self._analyze_air_displacement(frames),
            'atmospheric_lensing': self._analyze_atmospheric_lensing(frames),
            'turbulence_patterns': self._analyze_turbulence(frames),
            'pressure_waves': self._detect_pressure_waves(frames),
            'condensation_effects': self._detect_condensation(frames)
        }
        
        return results
    
    def _analyze_heat_distortion(self, frames):
        """Detect heat shimmer and thermal distortion patterns."""
        distortion_maps = []
        
        for i in range(1, len(frames)):
            prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate local standard deviation to detect shimmer
            kernel = np.ones((5,5), np.float32) / 25
            mean_prev = cv2.filter2D(prev_frame.astype(np.float32), -1, kernel)
            mean_curr = cv2.filter2D(curr_frame.astype(np.float32), -1, kernel)
            
            # Detect rapid local variations
            diff = np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32))
            shimmer = cv2.GaussianBlur(diff, (3,3), 0)
            
            # High-frequency distortion detection
            laplacian = cv2.Laplacian(shimmer, cv2.CV_64F)
            distortion_map = np.abs(laplacian)
            
            distortion_maps.append(distortion_map)
        
        # Find areas of consistent distortion
        avg_distortion = np.mean(distortion_maps, axis=0)
        threshold = np.percentile(avg_distortion, 95)
        distortion_regions = avg_distortion > threshold
        
        return {
            'distortion_maps': distortion_maps,
            'avg_distortion': avg_distortion,
            'distortion_regions': distortion_regions,
            'distortion_intensity': float(np.mean(avg_distortion))
        }
    
    def _analyze_air_displacement(self, frames):
        """Detect air displacement through particle tracking."""
        displacement_vectors = []
        
        # Track small particles, dust, or debris
        detector = cv2.goodFeaturesToTrack
        lk_params = dict(winSize=(15,15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_pts = detector(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_pts is not None and len(prev_pts) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
                
                # Calculate displacement vectors
                good_new = next_pts[status==1]
                good_old = prev_pts[status==1]
                
                if len(good_new) > 0:
                    displacement = good_new - good_old
                    displacement_vectors.append(displacement)
                
                prev_pts = good_new.reshape(-1,1,2)
            else:
                prev_pts = detector(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            prev_gray = gray
        
        return {
            'displacement_vectors': displacement_vectors,
            'avg_displacement': np.mean([np.mean(d, axis=0) for d in displacement_vectors if len(d) > 0], axis=0) if displacement_vectors else np.array([0, 0])
        }
    
    def _analyze_atmospheric_lensing(self, frames):
        """Detect atmospheric lensing effects that bend light."""
        lensing_effects = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect circular/elliptical distortions
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=5, maxRadius=100)
            
            # Analyze gradient discontinuities
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Find sudden gradient changes (lensing boundaries)
            grad_diff = ndimage.laplace(magnitude)
            lensing_candidates = np.abs(grad_diff) > np.percentile(np.abs(grad_diff), 99)
            
            lensing_effects.append({
                'circles': circles,
                'gradient_magnitude': magnitude,
                'lensing_candidates': lensing_candidates,
                'lensing_strength': float(np.sum(lensing_candidates))
            })
        
        return lensing_effects
    
    def _analyze_turbulence(self, frames):
        """Detect atmospheric turbulence patterns."""
        turbulence_data = []
        
        for i in range(1, len(frames)):
            prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Analyze flow divergence and curl
            fx, fy = flow[..., 0], flow[..., 1]
            
            # Calculate divergence (expansion/compression)
            fx_dx = np.gradient(fx, axis=1)
            fy_dy = np.gradient(fy, axis=0)
            divergence = fx_dx + fy_dy
            
            # Calculate curl (rotation)
            fy_dx = np.gradient(fy, axis=1)
            fx_dy = np.gradient(fx, axis=0)
            curl = fy_dx - fx_dy
            
            # Detect vortex structures
            vorticity = np.abs(curl)
            vortex_threshold = np.percentile(vorticity, 95)
            vortex_regions = vorticity > vortex_threshold
            
            turbulence_data.append({
                'divergence': divergence,
                'curl': curl,
                'vorticity': vorticity,
                'vortex_regions': vortex_regions,
                'turbulence_intensity': float(np.std(vorticity))
            })
        
        return turbulence_data
    
    def _detect_pressure_waves(self, frames):
        """Detect pressure waves through brightness oscillations."""
        brightness_data = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Calculate local brightness variations
            brightness_data.append(np.mean(gray))
        
        # Look for periodic pressure wave signatures
        brightness_array = np.array(brightness_data)
        
        # Detrend the signal
        detrended = brightness_array - np.mean(brightness_array)
        
        # Find oscillations
        peaks, _ = find_peaks(detrended, height=np.std(detrended))
        valleys, _ = find_peaks(-detrended, height=np.std(detrended))
        
        # Calculate wave frequency
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            avg_interval = np.mean(peak_intervals)
            fps = 30  # Default, should come from metadata
            frequency = fps / avg_interval if avg_interval > 0 else 0
        else:
            frequency = 0
        
        return {
            'brightness_data': brightness_data,
            'detrended_signal': detrended,
            'peaks': peaks,
            'valleys': valleys,
            'wave_frequency': frequency,
            'pressure_wave_detected': frequency > 0.1  # Threshold for detection
        }
    
    def _detect_condensation(self, frames):
        """Detect condensation clouds or vapor formation."""
        condensation_events = []
        
        for i, frame in enumerate(frames):
            # Convert to HSV for better cloud detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Detect white/light colored regions (clouds/vapor)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Detect sudden appearance of white regions
            if i > 0:
                prev_hsv = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2HSV)
                prev_white_mask = cv2.inRange(prev_hsv, lower_white, upper_white)
                
                # Find new white regions
                new_condensation = cv2.bitwise_and(white_mask, cv2.bitwise_not(prev_white_mask))
                
                # Analyze size and shape of condensation
                contours, _ = cv2.findContours(new_condensation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                significant_condensation = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:  # Minimum area threshold
                        # Calculate circularity
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        
                        significant_condensation.append({
                            'area': area,
                            'circularity': circularity,
                            'contour': contour
                        })
                
                if significant_condensation:
                    condensation_events.append({
                        'frame': i,
                        'condensation_regions': significant_condensation,
                        'total_area': sum(c['area'] for c in significant_condensation)
                    })
        
        return {
            'condensation_events': condensation_events,
            'event_count': len(condensation_events)
        }