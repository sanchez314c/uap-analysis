#!/usr/bin/env python3
"""
Multi-Spectral Analysis Component
================================

Analyzes multiple spectral bands including infrared, ultraviolet, and other
electromagnetic spectrum data when available, or infers spectral characteristics
from visible light video.
"""

import cv2
import numpy as np
from scipy import signal, ndimage
import logging

logger = logging.getLogger(__name__)

class MultiSpectralAnalyzer:
    """Multi-spectral and electromagnetic spectrum analysis."""
    
    def __init__(self, config):
        """Initialize multi-spectral analyzer."""
        self.config = config
        self.spectral_bands = {
            'visible': (380, 750),      # nm
            'near_ir': (750, 1400),     # nm
            'thermal_ir': (3000, 12000), # nm
            'uv': (10, 380),            # nm
            'radio': (1e6, 1e12)        # nm (very rough conversion)
        }
        
    def analyze(self, frames, metadata, spectral_data=None):
        """Analyze multi-spectral characteristics."""
        logger.info("Starting multi-spectral analysis...")
        
        if spectral_data is not None:
            # Direct multi-spectral data analysis
            results = self._analyze_direct_spectral_data(spectral_data, metadata)
        else:
            # Infer spectral characteristics from visible light video
            results = self._analyze_inferred_spectral(frames, metadata)
        
        # Add specialized spectral analyses
        results.update({
            'thermal_signature_analysis': self._analyze_thermal_signatures(frames, spectral_data),
            'infrared_analysis': self._analyze_infrared_characteristics(frames, spectral_data),
            'ultraviolet_analysis': self._analyze_ultraviolet_signatures(frames, spectral_data),
            'electromagnetic_spectrum_analysis': self._analyze_em_spectrum(frames, spectral_data),
            'blackbody_radiation_analysis': self._analyze_blackbody_radiation(frames, spectral_data),
            'spectral_anomaly_detection': self._detect_spectral_anomalies(frames, spectral_data),
            'polarization_analysis': self._analyze_polarization_effects(frames),
            'spectral_line_analysis': self._analyze_spectral_lines(frames, spectral_data)
        })
        
        return results
    
    def _analyze_direct_spectral_data(self, spectral_data, metadata):
        """Analyze direct multi-spectral data."""
        spectral_analysis = {
            'spectral_bands_detected': self._identify_spectral_bands(spectral_data),
            'spectral_energy_distribution': self._analyze_energy_distribution(spectral_data),
            'spectral_peaks': self._identify_spectral_peaks(spectral_data),
            'spectral_classification': self._classify_spectral_signature(spectral_data),
            'temperature_estimation': self._estimate_temperature_from_spectrum(spectral_data),
            'material_identification': self._identify_materials_from_spectrum(spectral_data)
        }
        
        return spectral_analysis
    
    def _analyze_inferred_spectral(self, frames, metadata):
        """Infer spectral characteristics from visible light video."""
        inferred_analysis = {
            'color_temperature_analysis': self._analyze_color_temperature(frames),
            'rgb_spectral_decomposition': self._decompose_rgb_spectrum(frames),
            'visible_spectrum_analysis': self._analyze_visible_spectrum(frames),
            'spectral_evolution': self._analyze_spectral_evolution(frames),
            'color_saturation_analysis': self._analyze_color_saturation(frames),
            'white_balance_analysis': self._analyze_white_balance_anomalies(frames)
        }
        
        return inferred_analysis
    
    def _analyze_thermal_signatures(self, frames, spectral_data):
        """Analyze thermal infrared signatures."""
        thermal_analysis = []
        
        for i, frame in enumerate(frames):
            # If we have actual thermal data
            if spectral_data and 'thermal' in spectral_data:
                thermal_frame = spectral_data['thermal'][i] if i < len(spectral_data['thermal']) else None
                if thermal_frame is not None:
                    thermal_analysis.append(self._analyze_thermal_frame_direct(thermal_frame, i))
                    continue
            
            # Infer thermal characteristics from visible light
            thermal_inference = self._infer_thermal_from_visible(frame, i)
            thermal_analysis.append(thermal_inference)
        
        return {
            'frame_thermal_analysis': thermal_analysis,
            'thermal_hotspots': self._identify_thermal_hotspots(thermal_analysis),
            'temperature_distribution': self._analyze_temperature_distribution(thermal_analysis),
            'thermal_anomalies': self._detect_thermal_anomalies(thermal_analysis),
            'heat_source_classification': self._classify_heat_sources(thermal_analysis)
        }
    
    def _analyze_infrared_characteristics(self, frames, spectral_data):
        """Analyze near and mid-infrared characteristics."""
        ir_analysis = []
        
        for i, frame in enumerate(frames):
            # Analyze infrared signatures
            if spectral_data and 'infrared' in spectral_data:
                ir_frame = spectral_data['infrared'][i] if i < len(spectral_data['infrared']) else None
                if ir_frame is not None:
                    ir_analysis.append(self._analyze_ir_frame_direct(ir_frame, i))
                    continue
            
            # Infer IR characteristics from visible spectrum
            ir_inference = self._infer_ir_from_visible(frame, i)
            ir_analysis.append(ir_inference)
        
        return {
            'ir_frame_analysis': ir_analysis,
            'ir_emission_patterns': self._analyze_ir_emission_patterns(ir_analysis),
            'ir_reflection_analysis': self._analyze_ir_reflections(ir_analysis),
            'ir_transparency_analysis': self._analyze_ir_transparency(ir_analysis),
            'ir_signature_classification': self._classify_ir_signatures(ir_analysis)
        }
    
    def _analyze_ultraviolet_signatures(self, frames, spectral_data):
        """Analyze ultraviolet signatures."""
        uv_analysis = []
        
        for i, frame in enumerate(frames):
            if spectral_data and 'ultraviolet' in spectral_data:
                uv_frame = spectral_data['ultraviolet'][i] if i < len(spectral_data['ultraviolet']) else None
                if uv_frame is not None:
                    uv_analysis.append(self._analyze_uv_frame_direct(uv_frame, i))
                    continue
            
            # Infer UV characteristics
            uv_inference = self._infer_uv_from_visible(frame, i)
            uv_analysis.append(uv_inference)
        
        return {
            'uv_frame_analysis': uv_analysis,
            'uv_fluorescence': self._detect_uv_fluorescence(uv_analysis),
            'corona_discharge': self._detect_corona_discharge(uv_analysis),
            'plasma_signatures': self._detect_plasma_uv_signatures(uv_analysis),
            'atmospheric_uv_effects': self._analyze_atmospheric_uv_effects(uv_analysis)
        }
    
    def _analyze_em_spectrum(self, frames, spectral_data):
        """Analyze broader electromagnetic spectrum characteristics."""
        em_analysis = {
            'radio_frequency_analysis': self._analyze_radio_frequency_signatures(spectral_data),
            'microwave_analysis': self._analyze_microwave_signatures(spectral_data),
            'x_ray_analysis': self._analyze_x_ray_signatures(spectral_data),
            'gamma_ray_analysis': self._analyze_gamma_ray_signatures(spectral_data),
            'electromagnetic_field_effects': self._analyze_em_field_effects(frames, spectral_data)
        }
        
        return em_analysis
    
    def _analyze_blackbody_radiation(self, frames, spectral_data):
        """Analyze blackbody radiation characteristics."""
        blackbody_analysis = []
        
        for i, frame in enumerate(frames):
            # Estimate temperature from color temperature
            color_temp = self._estimate_color_temperature(frame)
            
            # Theoretical blackbody curve
            wavelengths = np.linspace(380, 750, 100)  # Visible spectrum
            blackbody_curve = self._calculate_blackbody_curve(wavelengths, color_temp)
            
            # Compare with observed spectrum
            observed_spectrum = self._extract_spectrum_from_frame(frame)
            blackbody_fit = self._fit_blackbody_to_spectrum(observed_spectrum, wavelengths)
            
            blackbody_analysis.append({
                'frame': i,
                'estimated_temperature': color_temp,
                'blackbody_curve': blackbody_curve.tolist(),
                'blackbody_fit_quality': blackbody_fit,
                'temperature_confidence': self._calculate_temperature_confidence(blackbody_fit)
            })
        
        return {
            'blackbody_analysis': blackbody_analysis,
            'temperature_estimates': [analysis['estimated_temperature'] for analysis in blackbody_analysis],
            'blackbody_compliance': self._assess_blackbody_compliance(blackbody_analysis),
            'non_thermal_emission': self._detect_non_thermal_emission(blackbody_analysis)
        }
    
    def _detect_spectral_anomalies(self, frames, spectral_data):
        """Detect anomalies in spectral characteristics."""
        spectral_anomalies = []
        
        for i, frame in enumerate(frames):
            frame_anomalies = []
            
            # Unusual color combinations
            color_anomalies = self._detect_unusual_colors(frame)
            if color_anomalies:
                frame_anomalies.extend(color_anomalies)
            
            # Impossible spectral signatures
            impossible_signatures = self._detect_impossible_spectral_signatures(frame)
            if impossible_signatures:
                frame_anomalies.extend(impossible_signatures)
            
            # Spectral line anomalies
            if spectral_data:
                line_anomalies = self._detect_spectral_line_anomalies(spectral_data, i)
                if line_anomalies:
                    frame_anomalies.extend(line_anomalies)
            
            # Non-physical emission patterns
            non_physical = self._detect_non_physical_emission(frame)
            if non_physical:
                frame_anomalies.extend(non_physical)
            
            if frame_anomalies:
                spectral_anomalies.append({
                    'frame': i,
                    'anomalies': frame_anomalies,
                    'anomaly_count': len(frame_anomalies)
                })
        
        return {
            'spectral_anomalies': spectral_anomalies,
            'total_anomaly_count': sum(len(frame['anomalies']) for frame in spectral_anomalies),
            'anomaly_severity': self._calculate_spectral_anomaly_severity(spectral_anomalies),
            'anomaly_categories': self._categorize_spectral_anomalies(spectral_anomalies)
        }
    
    def _analyze_polarization_effects(self, frames):
        """Analyze polarization effects in the imagery."""
        polarization_analysis = []
        
        for i, frame in enumerate(frames):
            # Analyze potential polarization effects
            polarization_indicators = self._detect_polarization_indicators(frame)
            
            # Look for polarized reflections
            polarized_reflections = self._detect_polarized_reflections(frame)
            
            # Analyze scattering polarization
            scattering_polarization = self._analyze_scattering_polarization(frame)
            
            polarization_analysis.append({
                'frame': i,
                'polarization_indicators': polarization_indicators,
                'polarized_reflections': polarized_reflections,
                'scattering_polarization': scattering_polarization
            })
        
        return {
            'polarization_analysis': polarization_analysis,
            'polarization_detected': any(frame['polarization_indicators']['detected'] 
                                       for frame in polarization_analysis),
            'polarization_strength': self._calculate_average_polarization_strength(polarization_analysis)
        }
    
    def _analyze_spectral_lines(self, frames, spectral_data):
        """Analyze spectral emission and absorption lines."""
        if spectral_data is None:
            return {'error': 'No spectral data available for line analysis'}
        
        spectral_line_analysis = {
            'emission_lines': self._detect_emission_lines(spectral_data),
            'absorption_lines': self._detect_absorption_lines(spectral_data),
            'doppler_shifts': self._detect_doppler_shifts_spectral(spectral_data),
            'line_broadening': self._analyze_line_broadening(spectral_data),
            'element_identification': self._identify_elements_from_lines(spectral_data),
            'ionization_signatures': self._detect_ionization_signatures(spectral_data)
        }
        
        return spectral_line_analysis
    
    # Helper methods for spectral analysis
    def _analyze_color_temperature(self, frames):
        """Analyze color temperature characteristics."""
        color_temperatures = []
        
        for i, frame in enumerate(frames):
            # Calculate color temperature
            color_temp = self._estimate_color_temperature(frame)
            
            # Analyze color temperature distribution
            temp_distribution = self._analyze_color_temp_distribution(frame)
            
            color_temperatures.append({
                'frame': i,
                'color_temperature': color_temp,
                'temperature_distribution': temp_distribution,
                'temperature_uniformity': self._calculate_temp_uniformity(temp_distribution)
            })
        
        return {
            'color_temperatures': color_temperatures,
            'average_color_temperature': np.mean([ct['color_temperature'] for ct in color_temperatures]),
            'color_temperature_variation': np.std([ct['color_temperature'] for ct in color_temperatures]),
            'temperature_classification': self._classify_color_temperature([ct['color_temperature'] for ct in color_temperatures])
        }
    
    def _estimate_color_temperature(self, frame):
        """Estimate color temperature from RGB values."""
        # Convert to floating point
        frame_float = frame.astype(np.float32)
        
        # Calculate average RGB values
        avg_b = np.mean(frame_float[:, :, 0])
        avg_g = np.mean(frame_float[:, :, 1])
        avg_r = np.mean(frame_float[:, :, 2])
        
        # Avoid division by zero
        if avg_b == 0:
            avg_b = 1
        
        # Color temperature estimation using RGB ratios
        # This is a simplified approach
        blue_red_ratio = avg_b / avg_r if avg_r > 0 else 1
        
        # Empirical relationship (simplified)
        if blue_red_ratio > 1.2:
            color_temp = 6500 + (blue_red_ratio - 1.2) * 2000  # Cool
        elif blue_red_ratio < 0.8:
            color_temp = 3000 + blue_red_ratio * 1250  # Warm
        else:
            color_temp = 5500  # Neutral
        
        return float(np.clip(color_temp, 1000, 12000))
    
    def _calculate_blackbody_curve(self, wavelengths, temperature):
        """Calculate theoretical blackbody radiation curve."""
        # Planck's law
        h = 6.626e-34  # Planck constant
        c = 3e8        # Speed of light
        k = 1.381e-23  # Boltzmann constant
        
        wavelengths_m = wavelengths * 1e-9  # Convert nm to m
        
        # Planck function
        numerator = 2 * h * c**2 / (wavelengths_m**5)
        denominator = np.exp(h * c / (wavelengths_m * k * temperature)) - 1
        
        blackbody_intensity = numerator / denominator
        
        # Normalize
        blackbody_intensity = blackbody_intensity / np.max(blackbody_intensity)
        
        return blackbody_intensity
    
    def _extract_spectrum_from_frame(self, frame):
        """Extract spectrum from RGB frame."""
        # Simple RGB to spectrum approximation
        # This is a simplified approach
        b, g, r = cv2.split(frame)
        
        # Map RGB to wavelength ranges
        # Blue: ~450nm, Green: ~550nm, Red: ~650nm
        wavelengths = np.array([450, 550, 650])
        intensities = np.array([np.mean(b), np.mean(g), np.mean(r)])
        
        # Interpolate to get full spectrum
        full_wavelengths = np.linspace(380, 750, 100)
        full_spectrum = np.interp(full_wavelengths, wavelengths, intensities)
        
        return full_spectrum / np.max(full_spectrum)
    
    def _detect_unusual_colors(self, frame):
        """Detect unusual or impossible color combinations."""
        unusual_colors = []
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Analyze hue distribution
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Detect unusual high saturation with low value (shouldn't occur naturally)
        unusual_mask = (saturation > 200) & (value < 50)
        if np.sum(unusual_mask) > frame.size * 0.01:  # More than 1% of pixels
            unusual_colors.append({
                'type': 'high_saturation_low_value',
                'percentage': float(np.sum(unusual_mask) / frame.size * 100),
                'description': 'Highly saturated colors with low brightness'
            })
        
        # Detect impossible color combinations
        # Pure colors that are too intense
        max_rgb = np.max(frame, axis=2)
        pure_color_mask = max_rgb > 250
        if np.sum(pure_color_mask) > frame.size * 0.05:  # More than 5%
            unusual_colors.append({
                'type': 'oversaturated_colors',
                'percentage': float(np.sum(pure_color_mask) / frame.size * 100),
                'description': 'Oversaturated colors suggesting non-natural illumination'
            })
        
        return unusual_colors
    
    def _infer_thermal_from_visible(self, frame, frame_idx):
        """Infer thermal characteristics from visible light."""
        # Analyze red channel as thermal proxy
        red_channel = frame[:, :, 2]
        
        # Calculate thermal statistics
        thermal_stats = {
            'mean_thermal': float(np.mean(red_channel)),
            'max_thermal': float(np.max(red_channel)),
            'thermal_std': float(np.std(red_channel)),
            'thermal_hotspots': self._detect_red_hotspots(red_channel)
        }
        
        # Estimate temperature based on red intensity
        estimated_temp = 273 + (np.mean(red_channel) / 255) * 100  # Very rough estimate
        
        return {
            'frame': frame_idx,
            'thermal_statistics': thermal_stats,
            'estimated_temperature': float(estimated_temp),
            'thermal_confidence': 0.3  # Low confidence for inference
        }
    
    def _detect_red_hotspots(self, red_channel):
        """Detect potential thermal hotspots from red channel."""
        threshold = np.percentile(red_channel, 95)
        hotspot_mask = red_channel > threshold
        
        # Find connected components
        contours, _ = cv2.findContours(hotspot_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hotspots = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Minimum area
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    hotspots.append({
                        'center': [cx, cy],
                        'area': float(cv2.contourArea(contour)),
                        'intensity': float(np.mean(red_channel[hotspot_mask]))
                    })
        
        return hotspots
    
    def _detect_polarization_indicators(self, frame):
        """Detect indicators of polarized light."""
        # Look for patterns that might indicate polarization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze for uniform illumination patterns
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Look for directional patterns
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        # Analyze direction histogram
        direction_hist, bins = np.histogram(gradient_direction.flatten(), bins=36)
        
        # Strong peaks in specific directions might indicate polarization
        max_direction_strength = np.max(direction_hist) / np.sum(direction_hist)
        
        polarization_detected = max_direction_strength > 0.2  # Threshold
        
        return {
            'detected': polarization_detected,
            'strength': float(max_direction_strength),
            'dominant_direction': float(bins[np.argmax(direction_hist)]),
            'direction_histogram': direction_hist.tolist()
        }
    
    def _classify_color_temperature(self, color_temperatures):
        """Classify color temperature range."""
        avg_temp = np.mean(color_temperatures)
        
        if avg_temp < 3000:
            return 'very_warm'
        elif avg_temp < 4000:
            return 'warm'
        elif avg_temp < 5500:
            return 'neutral'
        elif avg_temp < 7000:
            return 'cool'
        else:
            return 'very_cool'