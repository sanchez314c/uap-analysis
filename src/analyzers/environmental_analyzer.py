#!/usr/bin/env python3
"""
Environmental Correlation Analysis Component
===========================================

Correlates UAP observations with environmental conditions including weather,
atmospheric conditions, electromagnetic environment, and temporal patterns.
"""

import cv2
import numpy as np
import json
from datetime import datetime, timezone
from scipy import signal
import logging

logger = logging.getLogger(__name__)

class EnvironmentalAnalyzer:
    """Analyzes environmental factors and their correlation with UAP observations."""
    
    def __init__(self, config):
        """Initialize environmental analyzer."""
        self.config = config
        self.weather_api_key = config.get('environmental', {}).get('weather_api_key')
        
    def analyze(self, frames, metadata):
        """Analyze environmental correlations."""
        logger.info("Starting environmental correlation analysis...")
        
        # Extract timestamp and location if available
        timestamp = metadata.get('timestamp') or self._extract_timestamp_from_metadata(metadata)
        location = metadata.get('location') or self._estimate_location_from_metadata(metadata)
        
        results = {
            'atmospheric_conditions': self._analyze_atmospheric_conditions(frames, timestamp, location),
            'weather_correlation': self._analyze_weather_correlation(timestamp, location),
            'electromagnetic_environment': self._analyze_em_environment(frames, timestamp, location),
            'temporal_patterns': self._analyze_temporal_patterns(timestamp),
            'celestial_correlation': self._analyze_celestial_correlation(timestamp, location),
            'atmospheric_opacity': self._analyze_atmospheric_opacity(frames),
            'environmental_anomalies': self._detect_environmental_anomalies(frames),
            'visibility_conditions': self._analyze_visibility_conditions(frames, timestamp, location)
        }
        
        # Calculate environmental risk factors
        results['environmental_score'] = self._calculate_environmental_score(results)
        
        return results
    
    def _analyze_atmospheric_conditions(self, frames, timestamp, location):
        """Analyze atmospheric conditions from video characteristics."""
        atmospheric_data = []
        
        for i, frame in enumerate(frames):
            # Analyze atmospheric clarity
            clarity = self._measure_atmospheric_clarity(frame)
            
            # Detect atmospheric layers/inversions
            layers = self._detect_atmospheric_layers(frame)
            
            # Analyze humidity indicators
            humidity_indicators = self._analyze_humidity_indicators(frame)
            
            # Detect atmospheric disturbances
            disturbances = self._detect_atmospheric_disturbances(frame)
            
            # Measure atmospheric turbulence
            turbulence = self._measure_atmospheric_turbulence(frame, frames[max(0, i-1)])
            
            atmospheric_data.append({
                'frame': i,
                'clarity_score': clarity,
                'atmospheric_layers': layers,
                'humidity_indicators': humidity_indicators,
                'disturbances': disturbances,
                'turbulence_level': turbulence
            })
        
        # Aggregate atmospheric analysis
        return {
            'frame_analysis': atmospheric_data,
            'average_clarity': np.mean([d['clarity_score'] for d in atmospheric_data]),
            'atmospheric_stability': self._calculate_atmospheric_stability(atmospheric_data),
            'inversion_layers_detected': any(d['atmospheric_layers']['inversion_detected'] for d in atmospheric_data),
            'atmospheric_quality_score': self._calculate_atmospheric_quality(atmospheric_data)
        }
    
    def _analyze_weather_correlation(self, timestamp, location):
        """Correlate with weather conditions."""
        weather_data = {}
        
        try:
            # Attempt to get historical weather data
            if timestamp and location:
                weather_data = self._fetch_historical_weather(timestamp, location)
            
            # Analyze weather patterns that correlate with UAP sightings
            weather_correlations = self._analyze_weather_patterns(weather_data)
            
            # Check for unusual weather conditions
            unusual_conditions = self._detect_unusual_weather(weather_data)
            
            return {
                'weather_data': weather_data,
                'correlations': weather_correlations,
                'unusual_conditions': unusual_conditions,
                'weather_score': self._calculate_weather_correlation_score(weather_data)
            }
            
        except Exception as e:
            logger.warning(f"Weather analysis failed: {e}")
            return {
                'error': 'Weather data unavailable',
                'fallback_analysis': self._fallback_weather_analysis()
            }
    
    def _analyze_em_environment(self, frames, timestamp, location):
        """Analyze electromagnetic environment."""
        em_analysis = []
        
        for i, frame in enumerate(frames):
            # Analyze EM interference in video
            em_noise = self._detect_em_interference(frame)
            
            # Look for unusual electromagnetic signatures
            em_signatures = self._analyze_em_signatures(frame)
            
            # Detect electromagnetic anomalies
            em_anomalies = self._detect_em_anomalies(frame, frames[max(0, i-1)] if i > 0 else frame)
            
            em_analysis.append({
                'frame': i,
                'em_interference': em_noise,
                'em_signatures': em_signatures,
                'em_anomalies': em_anomalies
            })
        
        # Correlate with known EM sources
        em_sources = self._correlate_em_sources(location, timestamp)
        
        return {
            'frame_analysis': em_analysis,
            'em_source_correlation': em_sources,
            'em_anomaly_score': self._calculate_em_anomaly_score(em_analysis),
            'interference_patterns': self._analyze_interference_patterns(em_analysis)
        }
    
    def _analyze_temporal_patterns(self, timestamp):
        """Analyze temporal patterns and correlations."""
        if not timestamp:
            return {'error': 'No timestamp available'}
        
        try:
            dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
        except:
            return {'error': 'Invalid timestamp format'}
        
        temporal_analysis = {
            'time_of_day': self._categorize_time_of_day(dt),
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'season': self._get_season(dt),
            'lunar_phase': self._get_lunar_phase(dt),
            'solar_activity': self._get_solar_activity_estimate(dt),
            'daylight_conditions': self._analyze_daylight_conditions(dt),
            'temporal_anomalies': self._detect_temporal_anomalies(dt)
        }
        
        # Statistical analysis of timing
        temporal_analysis['timing_significance'] = self._analyze_timing_significance(temporal_analysis)
        
        return temporal_analysis
    
    def _analyze_celestial_correlation(self, timestamp, location):
        """Analyze correlation with celestial events."""
        if not timestamp:
            return {'error': 'No timestamp for celestial analysis'}
        
        try:
            dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
        except:
            return {'error': 'Invalid timestamp format'}
        
        celestial_data = {
            'moon_phase': self._calculate_moon_phase(dt),
            'moon_position': self._estimate_moon_position(dt, location),
            'planet_positions': self._estimate_planet_positions(dt),
            'meteor_shower_correlation': self._check_meteor_showers(dt),
            'satellite_passes': self._estimate_satellite_activity(dt, location),
            'iss_correlation': self._check_iss_visibility(dt, location),
            'celestial_events': self._check_celestial_events(dt)
        }
        
        # Calculate celestial correlation score
        celestial_data['correlation_score'] = self._calculate_celestial_correlation_score(celestial_data)
        
        return celestial_data
    
    def _analyze_atmospheric_opacity(self, frames):
        """Analyze atmospheric opacity and visibility."""
        opacity_data = []
        
        for i, frame in enumerate(frames):
            # Measure overall frame contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            
            # Analyze histogram spread
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_spread = np.std(hist)
            
            # Detect haze/fog indicators
            haze_score = self._detect_haze(frame)
            
            # Measure visibility distance estimate
            visibility_estimate = self._estimate_visibility_distance(frame)
            
            opacity_data.append({
                'frame': i,
                'contrast': float(contrast),
                'histogram_spread': float(hist_spread),
                'haze_score': haze_score,
                'visibility_estimate': visibility_estimate
            })
        
        return {
            'opacity_analysis': opacity_data,
            'average_visibility': np.mean([d['visibility_estimate'] for d in opacity_data]),
            'atmospheric_clarity': np.mean([d['contrast'] for d in opacity_data]),
            'haze_presence': np.mean([d['haze_score'] for d in opacity_data])
        }
    
    def _detect_environmental_anomalies(self, frames):
        """Detect unusual environmental conditions."""
        anomalies = []
        
        for i, frame in enumerate(frames):
            # Detect unusual lighting conditions
            lighting_anomalies = self._detect_lighting_anomalies(frame)
            
            # Detect atmospheric anomalies
            atmospheric_anomalies = self._detect_unusual_atmospheric_effects(frame)
            
            # Detect color anomalies that might indicate unusual conditions
            color_anomalies = self._detect_color_anomalies(frame)
            
            if lighting_anomalies or atmospheric_anomalies or color_anomalies:
                anomalies.append({
                    'frame': i,
                    'lighting_anomalies': lighting_anomalies,
                    'atmospheric_anomalies': atmospheric_anomalies,
                    'color_anomalies': color_anomalies
                })
        
        return {
            'anomaly_frames': anomalies,
            'anomaly_count': len(anomalies),
            'anomaly_types': self._categorize_anomaly_types(anomalies)
        }
    
    def _analyze_visibility_conditions(self, frames, timestamp, location):
        """Analyze overall visibility conditions."""
        # Combine multiple visibility factors
        atmospheric_vis = self._analyze_atmospheric_opacity(frames)
        
        # Time-based visibility factors
        if timestamp:
            temporal_vis = self._analyze_temporal_visibility(timestamp)
        else:
            temporal_vis = {}
        
        # Frame-based visibility analysis
        frame_visibility = []
        for frame in frames:
            vis_score = self._calculate_frame_visibility_score(frame)
            frame_visibility.append(vis_score)
        
        return {
            'atmospheric_visibility': atmospheric_vis,
            'temporal_visibility': temporal_vis,
            'frame_visibility_scores': frame_visibility,
            'overall_visibility_score': np.mean(frame_visibility),
            'visibility_consistency': np.std(frame_visibility),
            'optimal_viewing_conditions': self._assess_optimal_conditions(atmospheric_vis, temporal_vis, frame_visibility)
        }
    
    def _calculate_environmental_score(self, results):
        """Calculate overall environmental correlation score."""
        score_factors = []
        
        # Atmospheric conditions score
        if 'atmospheric_conditions' in results:
            atm_score = results['atmospheric_conditions'].get('atmospheric_quality_score', 0.5)
            score_factors.append(('atmospheric', atm_score, 0.3))
        
        # Weather correlation score
        if 'weather_correlation' in results and 'weather_score' in results['weather_correlation']:
            weather_score = results['weather_correlation']['weather_score']
            score_factors.append(('weather', weather_score, 0.2))
        
        # EM environment score
        if 'electromagnetic_environment' in results:
            em_score = results['electromagnetic_environment'].get('em_anomaly_score', 0.5)
            score_factors.append(('electromagnetic', em_score, 0.2))
        
        # Temporal patterns score
        if 'temporal_patterns' in results and 'timing_significance' in results['temporal_patterns']:
            temporal_score = results['temporal_patterns']['timing_significance']
            score_factors.append(('temporal', temporal_score, 0.15))
        
        # Celestial correlation score
        if 'celestial_correlation' in results and 'correlation_score' in results['celestial_correlation']:
            celestial_score = results['celestial_correlation']['correlation_score']
            score_factors.append(('celestial', celestial_score, 0.15))
        
        # Calculate weighted score
        if score_factors:
            total_weight = sum(weight for _, _, weight in score_factors)
            weighted_score = sum(score * weight for _, score, weight in score_factors) / total_weight
        else:
            weighted_score = 0.5
        
        return {
            'overall_score': float(weighted_score),
            'contributing_factors': {name: score for name, score, _ in score_factors},
            'environmental_favorability': self._categorize_environmental_favorability(weighted_score)
        }
    
    # Helper methods for environmental analysis
    def _measure_atmospheric_clarity(self, frame):
        """Measure atmospheric clarity from frame characteristics."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (sharpness indicator)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Combine metrics
        clarity_score = (laplacian_var / 1000 + contrast / 100) / 2
        
        return min(clarity_score, 1.0)
    
    def _detect_atmospheric_layers(self, frame):
        """Detect atmospheric layers and inversions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze horizontal bands that might indicate layers
        horizontal_profile = np.mean(gray, axis=1)
        
        # Look for sudden changes in intensity (layer boundaries)
        gradient = np.gradient(horizontal_profile)
        peaks, _ = signal.find_peaks(np.abs(gradient), height=np.std(gradient))
        
        # Detect temperature inversion indicators
        inversion_detected = len(peaks) > 2 and np.std(horizontal_profile) > 20
        
        return {
            'layer_boundaries': peaks.tolist(),
            'layer_count': len(peaks),
            'inversion_detected': inversion_detected,
            'intensity_variation': float(np.std(horizontal_profile))
        }
    
    def _analyze_humidity_indicators(self, frame):
        """Analyze visual indicators of humidity."""
        # Convert to HSV for better analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Look for haze/moisture indicators
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Low saturation + high value often indicates haze
        haze_mask = (saturation < 50) & (value > 150)
        haze_percentage = np.sum(haze_mask) / haze_mask.size
        
        # Analyze color temperature shifts due to moisture
        blue_channel = frame[:, :, 0]
        red_channel = frame[:, :, 2]
        color_temp_shift = np.mean(blue_channel) - np.mean(red_channel)
        
        return {
            'haze_percentage': float(haze_percentage),
            'color_temperature_shift': float(color_temp_shift),
            'humidity_estimate': min(haze_percentage * 2, 1.0)
        }
    
    def _detect_atmospheric_disturbances(self, frame):
        """Detect atmospheric disturbances."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # High-frequency noise analysis
        high_freq_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(gray, cv2.CV_64F, high_freq_filter)
        
        disturbance_level = np.std(high_freq)
        
        # Detect rapid intensity changes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'disturbance_level': float(disturbance_level),
            'gradient_variance': float(np.var(gradient_magnitude)),
            'atmospheric_turbulence': disturbance_level > np.percentile([disturbance_level], 75)
        }
    
    def _measure_atmospheric_turbulence(self, current_frame, previous_frame):
        """Measure atmospheric turbulence between frames."""
        if current_frame.shape != previous_frame.shape:
            return 0.0
        
        # Calculate frame difference
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray_curr, gray_prev)
        
        # Focus on small-scale variations (turbulence)
        turbulence_kernel = np.ones((3, 3), np.float32) / 9
        smoothed_diff = cv2.filter2D(diff, -1, turbulence_kernel)
        
        turbulence_level = np.std(smoothed_diff)
        
        return float(turbulence_level)
    
    def _fetch_historical_weather(self, timestamp, location):
        """Fetch historical weather data (placeholder implementation)."""
        # This would integrate with weather APIs like OpenWeatherMap, etc.
        # For now, return placeholder data
        return {
            'temperature': 20.0,
            'humidity': 65.0,
            'pressure': 1013.25,
            'wind_speed': 5.0,
            'wind_direction': 180,
            'visibility': 10000,
            'cloud_cover': 0.3,
            'precipitation': 0.0,
            'weather_conditions': 'clear'
        }
    
    def _categorize_time_of_day(self, dt):
        """Categorize time of day."""
        hour = dt.hour
        
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def _get_season(self, dt):
        """Get season from date."""
        month = dt.month
        
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _get_lunar_phase(self, dt):
        """Get lunar phase (simplified calculation)."""
        # Simplified lunar phase calculation
        # Days since new moon reference (Jan 1, 2000)
        ref_date = datetime(2000, 1, 1)
        days_since_ref = (dt - ref_date).days
        
        # Lunar cycle is approximately 29.53 days
        lunar_cycle = 29.53
        phase_position = (days_since_ref % lunar_cycle) / lunar_cycle
        
        if phase_position < 0.125:
            return 'new_moon'
        elif phase_position < 0.375:
            return 'waxing_crescent'
        elif phase_position < 0.625:
            return 'full_moon'
        else:
            return 'waning_crescent'
    
    def _detect_haze(self, frame):
        """Detect haze in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Haze typically reduces contrast and increases overall brightness
        contrast = np.std(gray)
        brightness = np.mean(gray)
        
        # Normalized haze score
        haze_score = (brightness / 255) * (1 - contrast / 128)
        
        return float(min(haze_score, 1.0))
    
    def _estimate_visibility_distance(self, frame):
        """Estimate visibility distance from frame characteristics."""
        # Simplified visibility estimation based on contrast and clarity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        contrast = np.std(gray)
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Empirical relationship (simplified)
        if contrast > 60 and clarity > 800:
            visibility = 50000  # 50km - excellent visibility
        elif contrast > 40 and clarity > 400:
            visibility = 20000  # 20km - good visibility
        elif contrast > 20 and clarity > 100:
            visibility = 5000   # 5km - moderate visibility
        else:
            visibility = 1000   # 1km - poor visibility
        
        return visibility
    
    def _detect_lighting_anomalies(self, frame):
        """Detect unusual lighting conditions."""
        # Analyze color temperature
        b, g, r = cv2.split(frame)
        
        # Calculate color temperature indicators
        blue_red_ratio = np.mean(b) / (np.mean(r) + 1)
        green_intensity = np.mean(g)
        
        anomalies = []
        
        # Check for unusual color casts
        if blue_red_ratio > 1.5:
            anomalies.append('unusual_blue_cast')
        elif blue_red_ratio < 0.5:
            anomalies.append('unusual_red_cast')
        
        # Check for unusual green tint
        if green_intensity > np.mean([np.mean(b), np.mean(r)]) * 1.3:
            anomalies.append('unusual_green_cast')
        
        # Check for extreme brightness variations
        brightness_std = np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if brightness_std > 80:
            anomalies.append('extreme_brightness_variation')
        
        return anomalies
    
    def _calculate_frame_visibility_score(self, frame):
        """Calculate visibility score for a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple visibility factors
        contrast = np.std(gray) / 128  # Normalized contrast
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000  # Normalized sharpness
        
        # Avoid saturation
        saturation_penalty = 0
        if np.mean(gray) > 240 or np.mean(gray) < 15:
            saturation_penalty = 0.3
        
        visibility_score = (contrast + sharpness) / 2 - saturation_penalty
        
        return float(max(0, min(1, visibility_score)))
    
    def _categorize_environmental_favorability(self, score):
        """Categorize environmental favorability for UAP observation."""
        if score > 0.8:
            return 'excellent'
        elif score > 0.6:
            return 'good'
        elif score > 0.4:
            return 'fair'
        elif score > 0.2:
            return 'poor'
        else:
            return 'very_poor'