#!/usr/bin/env python3
"""
Signature Analysis Component
===========================

Analyzes unique signatures and patterns that could indicate specific
propulsion methods, energy sources, or technological characteristics.
"""

import cv2
import numpy as np
from scipy import signal, fft
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)

class SignatureAnalyzer:
    """Analyzes electromagnetic, thermal, and other technological signatures."""
    
    def __init__(self, config):
        """Initialize signature analyzer."""
        self.config = config
        
    def analyze(self, frames, metadata):
        """Analyze technological signatures in video frames."""
        logger.info("Starting signature analysis...")
        
        results = {
            'electromagnetic_signature': self._analyze_em_signature(frames),
            'thermal_signature': self._analyze_thermal_signature(frames),
            'propulsion_signature': self._analyze_propulsion_signature(frames),
            'energy_signature': self._analyze_energy_signature(frames),
            'plasma_signature': self._analyze_plasma_signature(frames),
            'field_signature': self._analyze_field_signature(frames),
            'harmonic_signature': self._analyze_harmonic_signature(frames),
            'interference_patterns': self._analyze_interference_patterns(frames)
        }
        
        # Calculate confidence scores
        results['signature_confidence'] = self._calculate_signature_confidence(results)
        
        return results
    
    def _analyze_em_signature(self, frames):
        """Analyze electromagnetic interference patterns."""
        em_signatures = []
        
        for i, frame in enumerate(frames):
            # Analyze noise patterns that might indicate EM interference
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # High-frequency noise analysis
            # Apply high-pass filter to isolate noise
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise = cv2.filter2D(gray, cv2.CV_64F, kernel)
            
            # Statistical analysis of noise
            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))
            
            # Frequency domain analysis
            f_transform = fft.fft2(gray)
            magnitude_spectrum = np.abs(f_transform)
            
            # Look for periodic interference patterns
            peaks = self._find_frequency_peaks(magnitude_spectrum)
            
            # Analyze banding patterns (common in EM interference)
            horizontal_bands = self._detect_banding(gray, axis=0)
            vertical_bands = self._detect_banding(gray, axis=1)
            
            em_signatures.append({
                'frame': i,
                'noise_std': float(noise_std),
                'noise_mean': float(noise_mean),
                'frequency_peaks': peaks,
                'horizontal_bands': horizontal_bands,
                'vertical_bands': vertical_bands,
                'em_anomaly_score': float(noise_std / (np.mean(gray) + 1))
            })
        
        return em_signatures
    
    def _analyze_thermal_signature(self, frames):
        """Analyze thermal emission patterns."""
        thermal_signatures = []
        
        for i, frame in enumerate(frames):
            # Convert to different color spaces for thermal analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Analyze heat patterns through color temperature
            # Look for infrared-like signatures in visible spectrum
            
            # Red channel analysis (often correlates with heat)
            red_channel = frame[:, :, 2]
            
            # Look for hot spots
            hot_spots = self._detect_hot_spots(red_channel)
            
            # Analyze heat distribution patterns
            heat_gradient = self._analyze_heat_gradient(red_channel)
            
            # Look for heat shimmer effects
            shimmer_score = self._detect_heat_shimmer(frame, frames[max(0, i-1)])
            
            thermal_signatures.append({
                'frame': i,
                'hot_spots': hot_spots,
                'heat_gradient': heat_gradient,
                'shimmer_score': shimmer_score,
                'thermal_intensity': float(np.mean(red_channel))
            })
        
        return thermal_signatures
    
    def _analyze_propulsion_signature(self, frames):
        """Analyze potential propulsion system signatures."""
        propulsion_signatures = []
        
        for i in range(1, len(frames)):
            current_frame = frames[i]
            previous_frame = frames[i-1]
            
            # Look for exhaust or emission patterns
            exhaust_patterns = self._detect_exhaust_patterns(current_frame, previous_frame)
            
            # Analyze directional energy emissions
            energy_vectors = self._analyze_energy_vectors(current_frame)
            
            # Look for field effects around the object
            field_effects = self._detect_field_effects(current_frame)
            
            # Analyze pulsing patterns that might indicate propulsion cycles
            pulsing_analysis = self._analyze_propulsion_pulsing(current_frame, previous_frame)
            
            propulsion_signatures.append({
                'frame': i,
                'exhaust_patterns': exhaust_patterns,
                'energy_vectors': energy_vectors,
                'field_effects': field_effects,
                'pulsing_analysis': pulsing_analysis
            })
        
        return propulsion_signatures
    
    def _analyze_energy_signature(self, frames):
        """Analyze energy emission and absorption patterns."""
        energy_signatures = []
        
        for frame in frames:
            # Analyze overall energy distribution
            energy_map = self._create_energy_map(frame)
            
            # Look for energy concentration points
            energy_centers = self._find_energy_centers(energy_map)
            
            # Analyze energy flow patterns
            energy_flow = self._analyze_energy_flow(frame)
            
            # Look for energy absorption patterns (dark areas with specific characteristics)
            absorption_patterns = self._detect_energy_absorption(frame)
            
            energy_signatures.append({
                'energy_map': energy_map,
                'energy_centers': energy_centers,
                'energy_flow': energy_flow,
                'absorption_patterns': absorption_patterns,
                'total_energy': float(np.sum(energy_map))
            })
        
        return energy_signatures
    
    def _analyze_plasma_signature(self, frames):
        """Analyze potential plasma or ionization signatures."""
        plasma_signatures = []
        
        for frame in frames:
            # Look for plasma-like characteristics
            # High intensity with specific color characteristics
            
            # Convert to HSV for better plasma detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Plasma often appears as bright white/blue regions
            plasma_mask = self._create_plasma_mask(hsv)
            
            # Analyze plasma density and distribution
            plasma_regions = self._analyze_plasma_regions(plasma_mask)
            
            # Look for characteristic plasma oscillations
            plasma_oscillations = self._detect_plasma_oscillations(frame)
            
            # Analyze spectral characteristics that might indicate ionization
            spectral_analysis = self._analyze_plasma_spectrum(frame)
            
            plasma_signatures.append({
                'plasma_mask': plasma_mask,
                'plasma_regions': plasma_regions,
                'plasma_oscillations': plasma_oscillations,
                'spectral_analysis': spectral_analysis,
                'plasma_intensity': float(np.sum(plasma_mask))
            })
        
        return plasma_signatures
    
    def _analyze_field_signature(self, frames):
        """Analyze potential field effects (magnetic, electric, gravitational)."""
        field_signatures = []
        
        for i, frame in enumerate(frames):
            # Look for field distortion patterns
            distortion_map = self._detect_field_distortions(frame)
            
            # Analyze radial patterns that might indicate field emanation
            radial_patterns = self._detect_radial_patterns(frame)
            
            # Look for lensing effects
            lensing_effects = self._detect_gravitational_lensing(frame)
            
            # Analyze particle deflection patterns (if particles visible)
            if i > 0:
                deflection_patterns = self._analyze_particle_deflection(frames[i-1], frame)
            else:
                deflection_patterns = None
            
            field_signatures.append({
                'frame': i,
                'distortion_map': distortion_map,
                'radial_patterns': radial_patterns,
                'lensing_effects': lensing_effects,
                'deflection_patterns': deflection_patterns
            })
        
        return field_signatures
    
    def _analyze_harmonic_signature(self, frames):
        """Analyze harmonic patterns and resonance signatures."""
        # Extract brightness data over time
        brightness_series = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_series.append(np.mean(gray))
        
        brightness_array = np.array(brightness_series)
        
        # Frequency domain analysis
        fft_result = fft.fft(brightness_array)
        frequencies = fft.fftfreq(len(brightness_array))
        magnitude = np.abs(fft_result)
        
        # Find dominant frequencies
        dominant_frequencies = self._find_dominant_frequencies(frequencies, magnitude)
        
        # Look for harmonic relationships
        harmonic_analysis = self._analyze_harmonic_relationships(dominant_frequencies)
        
        # Analyze phase relationships
        phase_analysis = self._analyze_phase_relationships(fft_result)
        
        return {
            'brightness_series': brightness_series,
            'fft_magnitude': magnitude.tolist(),
            'frequencies': frequencies.tolist(),
            'dominant_frequencies': dominant_frequencies,
            'harmonic_analysis': harmonic_analysis,
            'phase_analysis': phase_analysis
        }
    
    def _analyze_interference_patterns(self, frames):
        """Analyze interference patterns that might indicate wave interactions."""
        interference_patterns = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Look for wave interference patterns
            wave_patterns = self._detect_wave_patterns(gray)
            
            # Analyze constructive and destructive interference
            interference_analysis = self._analyze_interference_zones(gray)
            
            # Look for Moiré patterns
            moire_patterns = self._detect_moire_patterns(gray)
            
            # Analyze periodic structures
            periodic_analysis = self._analyze_periodic_structures(gray)
            
            interference_patterns.append({
                'wave_patterns': wave_patterns,
                'interference_analysis': interference_analysis,
                'moire_patterns': moire_patterns,
                'periodic_analysis': periodic_analysis
            })
        
        return interference_patterns
    
    def _calculate_signature_confidence(self, results):
        """Calculate confidence scores for different signature types."""
        confidence_scores = {}
        
        # EM signature confidence
        em_data = results.get('electromagnetic_signature', [])
        if em_data:
            avg_em_anomaly = np.mean([sig['em_anomaly_score'] for sig in em_data])
            confidence_scores['electromagnetic'] = min(avg_em_anomaly * 2, 1.0)
        
        # Thermal signature confidence
        thermal_data = results.get('thermal_signature', [])
        if thermal_data:
            hot_spot_count = sum(len(sig['hot_spots']) for sig in thermal_data)
            confidence_scores['thermal'] = min(hot_spot_count / len(thermal_data) * 0.2, 1.0)
        
        # Plasma signature confidence
        plasma_data = results.get('plasma_signature', [])
        if plasma_data:
            avg_plasma_intensity = np.mean([sig['plasma_intensity'] for sig in plasma_data])
            confidence_scores['plasma'] = min(avg_plasma_intensity / 1000, 1.0)
        
        # Overall confidence
        if confidence_scores:
            confidence_scores['overall'] = np.mean(list(confidence_scores.values()))
        else:
            confidence_scores['overall'] = 0.0
        
        return confidence_scores
    
    # Helper methods for signature analysis
    def _find_frequency_peaks(self, magnitude_spectrum):
        """Find peaks in frequency spectrum."""
        # Flatten and find peaks
        flat_spectrum = magnitude_spectrum.flatten()
        peaks, _ = signal.find_peaks(flat_spectrum, height=np.percentile(flat_spectrum, 99))
        return peaks.tolist()
    
    def _detect_banding(self, image, axis=0):
        """Detect banding patterns in image."""
        profile = np.mean(image, axis=axis)
        peaks, _ = signal.find_peaks(profile, distance=5)
        return len(peaks)
    
    def _detect_hot_spots(self, red_channel):
        """Detect thermal hot spots."""
        threshold = np.percentile(red_channel, 95)
        hot_spots = red_channel > threshold
        contours, _ = cv2.findContours(hot_spots.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hot_spot_data = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Minimum area
                center = np.mean(contour, axis=0)[0]
                hot_spot_data.append({
                    'center': center.tolist(),
                    'area': float(area)
                })
        
        return hot_spot_data
    
    def _analyze_heat_gradient(self, red_channel):
        """Analyze heat gradient patterns."""
        grad_x = cv2.Sobel(red_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(red_channel, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'max_gradient': float(np.max(magnitude)),
            'avg_gradient': float(np.mean(magnitude)),
            'gradient_concentration': float(np.std(magnitude))
        }
    
    def _detect_heat_shimmer(self, current_frame, previous_frame):
        """Detect heat shimmer effects."""
        if current_frame.shape != previous_frame.shape:
            return 0.0
        
        # Calculate pixel-wise differences
        diff = cv2.absdiff(current_frame, previous_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Look for high-frequency variations
        shimmer_score = np.std(gray_diff) / (np.mean(gray_diff) + 1)
        return float(shimmer_score)
    
    def _detect_exhaust_patterns(self, current_frame, previous_frame):
        """Detect exhaust or emission patterns."""
        # Look for directional bright regions that might indicate exhaust
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Find bright regions that appear suddenly
        diff = gray_current.astype(np.float32) - gray_previous.astype(np.float32)
        bright_appearances = diff > np.percentile(diff, 95)
        
        # Analyze shape and direction of bright regions
        contours, _ = cv2.findContours(bright_appearances.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        exhaust_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 20:
                # Calculate aspect ratio and orientation
                rect = cv2.minAreaRect(contour)
                aspect_ratio = max(rect[1]) / min(rect[1]) if min(rect[1]) > 0 else 0
                
                if aspect_ratio > 2:  # Elongated shapes might be exhaust
                    exhaust_candidates.append({
                        'area': float(area),
                        'aspect_ratio': float(aspect_ratio),
                        'angle': float(rect[2])
                    })
        
        return exhaust_candidates
    
    def _analyze_energy_vectors(self, frame):
        """Analyze directional energy patterns."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient vectors
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate energy flow directions
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Find dominant flow directions
        direction_hist, bins = np.histogram(direction.flatten(), bins=36)
        dominant_direction = bins[np.argmax(direction_hist)]
        
        return {
            'dominant_direction': float(dominant_direction),
            'energy_concentration': float(np.std(magnitude)),
            'max_energy_flow': float(np.max(magnitude))
        }
    
    def _detect_field_effects(self, frame):
        """Detect field-like distortion effects."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for circular or radial distortion patterns
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=10, maxRadius=100)
        
        # Analyze for concentric patterns
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            field_candidates = []
            
            for (x, y, r) in circles:
                # Analyze radial intensity pattern
                radial_profile = self._extract_radial_profile(gray, (x, y), r)
                field_candidates.append({
                    'center': [int(x), int(y)],
                    'radius': int(r),
                    'radial_profile': radial_profile
                })
            
            return field_candidates
        
        return []
    
    def _extract_radial_profile(self, image, center, max_radius):
        """Extract radial intensity profile."""
        x, y = center
        profile = []
        
        for r in range(1, min(max_radius, 50)):
            # Sample points at radius r
            angles = np.linspace(0, 2*np.pi, 16)
            sample_x = x + r * np.cos(angles)
            sample_y = y + r * np.sin(angles)
            
            # Ensure points are within image bounds
            valid_mask = ((sample_x >= 0) & (sample_x < image.shape[1]) & 
                         (sample_y >= 0) & (sample_y < image.shape[0]))
            
            if np.any(valid_mask):
                valid_x = sample_x[valid_mask].astype(int)
                valid_y = sample_y[valid_mask].astype(int)
                avg_intensity = np.mean(image[valid_y, valid_x])
                profile.append(avg_intensity)
        
        return profile
    
    def _create_energy_map(self, frame):
        """Create energy density map."""
        # Combine multiple color channels weighted by intensity
        b, g, r = cv2.split(frame)
        
        # Weight channels by typical energy association
        energy_map = 0.3 * b + 0.4 * g + 0.7 * r  # Red often indicates higher energy
        
        # Apply Gaussian blur to create energy field representation
        energy_map = cv2.GaussianBlur(energy_map, (5, 5), 0)
        
        return energy_map
    
    def _find_energy_centers(self, energy_map):
        """Find energy concentration centers."""
        # Find local maxima
        threshold = np.percentile(energy_map, 90)
        energy_centers = energy_map > threshold
        
        # Find connected components
        contours, _ = cv2.findContours(energy_centers.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(contour)
                centers.append({
                    'center': [cx, cy],
                    'area': float(area),
                    'intensity': float(np.mean(energy_map[energy_centers]))
                })
        
        return centers
    
    def _create_plasma_mask(self, hsv_image):
        """Create mask for potential plasma regions."""
        # Plasma typically appears as high-saturation, high-value regions
        h, s, v = cv2.split(hsv_image)
        
        # High value (brightness) and moderate to high saturation
        plasma_mask = (v > 200) & (s > 50)
        
        return plasma_mask.astype(np.uint8) * 255
    
    def _analyze_plasma_regions(self, plasma_mask):
        """Analyze plasma region characteristics."""
        contours, _ = cv2.findContours(plasma_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plasma_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                plasma_regions.append({
                    'area': float(area),
                    'circularity': float(circularity)
                })
        
        return plasma_regions
    
    # Additional helper methods would continue here...