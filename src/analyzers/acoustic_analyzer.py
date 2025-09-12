#!/usr/bin/env python3
"""
Acoustic Analysis Component
===========================

Analyzes audio signatures and sound patterns that may accompany UAP sightings.
Includes sonic boom detection, propulsion signatures, and environmental audio analysis.
"""

import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import logging

logger = logging.getLogger(__name__)

class AcousticAnalyzer:
    """Analyzes acoustic signatures and sound patterns."""
    
    def __init__(self, config):
        """Initialize acoustic analyzer."""
        self.config = config
        self.sample_rate = config.get('audio', {}).get('sample_rate', 44100)
        self.sonic_boom_threshold = config.get('acoustic', {}).get('sonic_boom_threshold', 0.8)
        
    def analyze(self, frames, metadata, audio_data=None):
        """Analyze acoustic signatures."""
        logger.info("Starting acoustic analysis...")
        
        if audio_data is not None:
            # Direct audio analysis
            results = self._analyze_direct_audio(audio_data, metadata)
        else:
            # Infer acoustic properties from video
            results = self._analyze_inferred_acoustics(frames, metadata)
        
        # Add specialized acoustic analyses
        results.update({
            'sonic_boom_analysis': self._analyze_sonic_boom_indicators(frames, audio_data),
            'propulsion_signatures': self._analyze_propulsion_acoustics(frames, audio_data),
            'environmental_audio': self._analyze_environmental_audio(frames, audio_data, metadata),
            'doppler_effect_analysis': self._analyze_doppler_effects(frames, audio_data, metadata),
            'acoustic_anomalies': self._detect_acoustic_anomalies(frames, audio_data),
            'silence_analysis': self._analyze_unusual_silence(frames, audio_data),
            'frequency_signatures': self._analyze_frequency_signatures(audio_data)
        })
        
        return results
    
    def _analyze_direct_audio(self, audio_data, metadata):
        """Analyze direct audio data if available."""
        if audio_data is None:
            return {'error': 'No audio data available'}
        
        # Convert to numpy array if needed
        if isinstance(audio_data, list):
            audio_array = np.array(audio_data)
        else:
            audio_array = audio_data
        
        # Basic audio analysis
        audio_analysis = {
            'audio_statistics': self._calculate_audio_statistics(audio_array),
            'frequency_analysis': self._perform_frequency_analysis(audio_array),
            'amplitude_analysis': self._analyze_amplitude_patterns(audio_array),
            'spectral_analysis': self._perform_spectral_analysis(audio_array),
            'noise_analysis': self._analyze_background_noise(audio_array),
            'transient_analysis': self._detect_transient_events(audio_array)
        }
        
        return audio_analysis
    
    def _analyze_inferred_acoustics(self, frames, metadata):
        """Infer acoustic properties from visual cues."""
        inferred_analysis = {
            'visual_acoustic_indicators': self._detect_visual_acoustic_indicators(frames),
            'motion_based_acoustics': self._infer_acoustics_from_motion(frames, metadata),
            'environmental_acoustic_inference': self._infer_environmental_acoustics(frames),
            'object_acoustic_properties': self._infer_object_acoustic_properties(frames)
        }
        
        return inferred_analysis
    
    def _analyze_sonic_boom_indicators(self, frames, audio_data):
        """Analyze for sonic boom signatures."""
        sonic_analysis = {
            'visual_shock_waves': self._detect_visual_shock_waves(frames),
            'pressure_wave_indicators': self._detect_pressure_wave_indicators(frames),
            'atmospheric_disturbance': self._analyze_atmospheric_disturbance_for_sonic(frames)
        }
        
        if audio_data is not None:
            sonic_analysis.update({
                'audio_sonic_signature': self._detect_audio_sonic_boom(audio_data),
                'pressure_wave_audio': self._analyze_pressure_waves_audio(audio_data),
                'sonic_boom_probability': self._calculate_sonic_boom_probability(audio_data, frames)
            })
        
        return sonic_analysis
    
    def _analyze_propulsion_acoustics(self, frames, audio_data):
        """Analyze propulsion-related acoustic signatures."""
        propulsion_analysis = {
            'conventional_propulsion': self._detect_conventional_propulsion_audio(frames, audio_data),
            'jet_engine_signatures': self._detect_jet_signatures(audio_data),
            'propeller_signatures': self._detect_propeller_signatures(audio_data),
            'rotor_signatures': self._detect_rotor_signatures(audio_data),
            'unusual_propulsion': self._detect_unusual_propulsion_signatures(frames, audio_data)
        }
        
        return propulsion_analysis
    
    def _analyze_environmental_audio(self, frames, audio_data, metadata):
        """Analyze environmental audio context."""
        env_analysis = {
            'background_noise_level': self._measure_background_noise(audio_data),
            'wind_noise_analysis': self._analyze_wind_noise(frames, audio_data),
            'ambient_sound_analysis': self._analyze_ambient_sounds(audio_data),
            'audio_environment_classification': self._classify_audio_environment(audio_data),
            'acoustic_conditions': self._assess_acoustic_conditions(frames, audio_data, metadata)
        }
        
        return env_analysis
    
    def _analyze_doppler_effects(self, frames, audio_data, metadata):
        """Analyze Doppler effect patterns."""
        if audio_data is None:
            return {'error': 'No audio data for Doppler analysis'}
        
        doppler_analysis = {
            'frequency_shifts': self._detect_frequency_shifts(audio_data),
            'doppler_signatures': self._analyze_doppler_signatures(audio_data),
            'velocity_estimates': self._estimate_velocity_from_doppler(audio_data, frames),
            'approach_recession_analysis': self._analyze_approach_recession(audio_data, frames)
        }
        
        return doppler_analysis
    
    def _detect_acoustic_anomalies(self, frames, audio_data):
        """Detect unusual acoustic phenomena."""
        anomalies = []
        
        if audio_data is not None:
            # Unusual frequency signatures
            freq_anomalies = self._detect_frequency_anomalies(audio_data)
            if freq_anomalies:
                anomalies.extend(freq_anomalies)
            
            # Impossible acoustic signatures
            impossible_signatures = self._detect_impossible_acoustics(audio_data)
            if impossible_signatures:
                anomalies.extend(impossible_signatures)
            
            # Electromagnetic interference in audio
            em_interference = self._detect_em_audio_interference(audio_data)
            if em_interference:
                anomalies.append(em_interference)
        
        # Visual-audio inconsistencies
        av_inconsistencies = self._detect_audio_visual_inconsistencies(frames, audio_data)
        if av_inconsistencies:
            anomalies.extend(av_inconsistencies)
        
        return {
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'acoustic_anomaly_score': self._calculate_acoustic_anomaly_score(anomalies)
        }
    
    def _analyze_unusual_silence(self, frames, audio_data):
        """Analyze unusual silence patterns."""
        if audio_data is None:
            return {'error': 'No audio data for silence analysis'}
        
        silence_analysis = {
            'silence_periods': self._detect_silence_periods(audio_data),
            'unusual_quiet_zones': self._detect_unusual_quiet_zones(audio_data, frames),
            'expected_vs_actual_noise': self._compare_expected_actual_noise(frames, audio_data),
            'silence_anomaly_score': self._calculate_silence_anomaly_score(audio_data, frames)
        }
        
        return silence_analysis
    
    def _analyze_frequency_signatures(self, audio_data):
        """Analyze detailed frequency signatures."""
        if audio_data is None:
            return {'error': 'No audio data for frequency analysis'}
        
        # Perform FFT analysis
        fft_result = fft(audio_data)
        frequencies = fftfreq(len(audio_data), 1/self.sample_rate)
        
        # Analyze frequency bands
        freq_analysis = {
            'infrasound_analysis': self._analyze_infrasound(fft_result, frequencies),
            'ultrasound_analysis': self._analyze_ultrasound(fft_result, frequencies),
            'harmonic_analysis': self._analyze_harmonics(fft_result, frequencies),
            'resonance_frequencies': self._detect_resonance_frequencies(fft_result, frequencies),
            'spectral_peaks': self._identify_spectral_peaks(fft_result, frequencies),
            'frequency_modulation': self._analyze_frequency_modulation(audio_data)
        }
        
        return freq_analysis
    
    # Helper methods for acoustic analysis
    def _calculate_audio_statistics(self, audio_array):
        """Calculate basic audio statistics."""
        return {
            'rms_amplitude': float(np.sqrt(np.mean(audio_array**2))),
            'peak_amplitude': float(np.max(np.abs(audio_array))),
            'dynamic_range': float(np.max(audio_array) - np.min(audio_array)),
            'zero_crossing_rate': self._calculate_zero_crossing_rate(audio_array),
            'spectral_centroid': self._calculate_spectral_centroid(audio_array),
            'spectral_rolloff': self._calculate_spectral_rolloff(audio_array)
        }
    
    def _perform_frequency_analysis(self, audio_array):
        """Perform detailed frequency domain analysis."""
        # FFT analysis
        fft_result = fft(audio_array)
        frequencies = fftfreq(len(audio_array), 1/self.sample_rate)
        magnitude = np.abs(fft_result)
        
        # Find dominant frequencies
        peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.1)
        dominant_frequencies = frequencies[peaks]
        
        return {
            'dominant_frequencies': dominant_frequencies[:10].tolist(),  # Top 10
            'frequency_spectrum': magnitude.tolist(),
            'fundamental_frequency': self._find_fundamental_frequency(magnitude, frequencies),
            'bandwidth': self._calculate_bandwidth(magnitude, frequencies),
            'spectral_flux': self._calculate_spectral_flux(magnitude)
        }
    
    def _analyze_amplitude_patterns(self, audio_array):
        """Analyze amplitude patterns and envelope."""
        # Calculate envelope
        envelope = np.abs(signal.hilbert(audio_array))
        
        # Detect amplitude modulation
        am_frequency = self._detect_amplitude_modulation(envelope)
        
        # Analyze dynamics
        dynamics = {
            'attack_time': self._calculate_attack_time(envelope),
            'decay_time': self._calculate_decay_time(envelope),
            'sustain_level': self._calculate_sustain_level(envelope),
            'release_time': self._calculate_release_time(envelope)
        }
        
        return {
            'envelope': envelope.tolist(),
            'amplitude_modulation_frequency': am_frequency,
            'dynamics': dynamics,
            'amplitude_stability': float(np.std(envelope) / np.mean(envelope)) if np.mean(envelope) > 0 else 0
        }
    
    def _detect_visual_shock_waves(self, frames):
        """Detect visual indicators of shock waves."""
        shock_indicators = []
        
        for i, frame in enumerate(frames):
            # Look for cone-shaped distortions
            cone_patterns = self._detect_cone_patterns(frame)
            
            # Detect sudden atmospheric distortions
            atmospheric_distortions = self._detect_sudden_distortions(frame, frames[max(0, i-1)])
            
            # Look for Mach diamond patterns
            mach_diamonds = self._detect_mach_diamonds(frame)
            
            if cone_patterns or atmospheric_distortions or mach_diamonds:
                shock_indicators.append({
                    'frame': i,
                    'cone_patterns': cone_patterns,
                    'atmospheric_distortions': atmospheric_distortions,
                    'mach_diamonds': mach_diamonds
                })
        
        return shock_indicators
    
    def _detect_audio_sonic_boom(self, audio_data):
        """Detect sonic boom signature in audio."""
        if audio_data is None:
            return None
        
        # Sonic boom characteristics:
        # - Double crack pattern (N-wave)
        # - High amplitude
        # - Short duration
        # - Specific frequency content
        
        # Look for N-wave pattern
        n_wave_detected = self._detect_n_wave_pattern(audio_data)
        
        # Analyze for double crack
        double_crack = self._detect_double_crack(audio_data)
        
        # Check amplitude characteristics
        amplitude_check = self._check_sonic_boom_amplitude(audio_data)
        
        # Frequency analysis for sonic boom
        freq_analysis = self._analyze_sonic_boom_frequency(audio_data)
        
        sonic_boom_probability = 0.0
        if n_wave_detected:
            sonic_boom_probability += 0.4
        if double_crack:
            sonic_boom_probability += 0.3
        if amplitude_check:
            sonic_boom_probability += 0.2
        if freq_analysis['sonic_boom_freq_match']:
            sonic_boom_probability += 0.1
        
        return {
            'n_wave_detected': n_wave_detected,
            'double_crack_detected': double_crack,
            'amplitude_consistent': amplitude_check,
            'frequency_analysis': freq_analysis,
            'sonic_boom_probability': min(sonic_boom_probability, 1.0)
        }
    
    def _detect_conventional_propulsion_audio(self, frames, audio_data):
        """Detect conventional aircraft propulsion signatures."""
        if audio_data is None:
            return {'error': 'No audio data'}
        
        propulsion_signatures = {
            'jet_engine_detected': self._detect_jet_engine_audio(audio_data),
            'propeller_detected': self._detect_propeller_audio(audio_data),
            'helicopter_rotor_detected': self._detect_helicopter_audio(audio_data),
            'turbofan_detected': self._detect_turbofan_audio(audio_data),
            'rocket_engine_detected': self._detect_rocket_audio(audio_data)
        }
        
        # Calculate conventional propulsion probability
        conventional_indicators = sum(1 for detected in propulsion_signatures.values() 
                                    if isinstance(detected, dict) and detected.get('detected', False))
        
        propulsion_signatures['conventional_propulsion_probability'] = min(conventional_indicators / 3, 1.0)
        
        return propulsion_signatures
    
    def _detect_frequency_shifts(self, audio_data):
        """Detect frequency shifts over time (Doppler effect)."""
        # Sliding window FFT to track frequency changes
        window_size = len(audio_data) // 20  # 20 windows
        hop_size = window_size // 2
        
        frequency_tracks = []
        
        for i in range(0, len(audio_data) - window_size, hop_size):
            window = audio_data[i:i + window_size]
            fft_result = fft(window)
            frequencies = fftfreq(window_size, 1/self.sample_rate)
            magnitude = np.abs(fft_result)
            
            # Find peak frequency
            peak_idx = np.argmax(magnitude)
            peak_frequency = frequencies[peak_idx]
            
            frequency_tracks.append({
                'time': i / self.sample_rate,
                'peak_frequency': float(peak_frequency),
                'magnitude': float(magnitude[peak_idx])
            })
        
        # Analyze frequency trends
        frequencies = [track['peak_frequency'] for track in frequency_tracks]
        frequency_drift = np.diff(frequencies)
        
        return {
            'frequency_tracks': frequency_tracks,
            'frequency_drift': frequency_drift.tolist(),
            'doppler_detected': np.std(frequency_drift) > 10,  # Threshold for significant drift
            'max_frequency_shift': float(np.max(np.abs(frequency_drift))) if len(frequency_drift) > 0 else 0
        }
    
    def _detect_frequency_anomalies(self, audio_data):
        """Detect unusual frequency signatures."""
        anomalies = []
        
        # Perform FFT
        fft_result = fft(audio_data)
        frequencies = fftfreq(len(audio_data), 1/self.sample_rate)
        magnitude = np.abs(fft_result)
        
        # Check for impossible frequencies
        max_freq = np.max(frequencies[magnitude > np.max(magnitude) * 0.1])
        if max_freq > 20000:  # Above human hearing
            anomalies.append({
                'type': 'ultrasonic_signature',
                'frequency': float(max_freq),
                'description': 'Ultrasonic frequencies detected'
            })
        
        # Check for infrasound
        infrasound_mask = (frequencies > 0) & (frequencies < 20)
        if np.any(magnitude[infrasound_mask] > np.max(magnitude) * 0.5):
            infrasound_freq = frequencies[infrasound_mask][np.argmax(magnitude[infrasound_mask])]
            anomalies.append({
                'type': 'infrasound_signature',
                'frequency': float(infrasound_freq),
                'description': 'Strong infrasound component detected'
            })
        
        # Check for electromagnetic interference patterns
        em_frequencies = [50, 60, 120, 180]  # Common EM interference frequencies
        for em_freq in em_frequencies:
            freq_idx = np.argmin(np.abs(frequencies - em_freq))
            if magnitude[freq_idx] > np.max(magnitude) * 0.3:
                anomalies.append({
                    'type': 'electromagnetic_interference',
                    'frequency': float(em_freq),
                    'description': f'Strong {em_freq}Hz component (EM interference)'
                })
        
        return anomalies
    
    def _calculate_zero_crossing_rate(self, audio_array):
        """Calculate zero crossing rate."""
        zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
        return float(zero_crossings / len(audio_array))
    
    def _calculate_spectral_centroid(self, audio_array):
        """Calculate spectral centroid."""
        fft_result = fft(audio_array)
        frequencies = fftfreq(len(audio_array), 1/self.sample_rate)
        magnitude = np.abs(fft_result)
        
        # Only use positive frequencies
        positive_freq_mask = frequencies > 0
        frequencies = frequencies[positive_freq_mask]
        magnitude = magnitude[positive_freq_mask]
        
        if np.sum(magnitude) > 0:
            centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
        else:
            centroid = 0
        
        return float(centroid)
    
    def _detect_n_wave_pattern(self, audio_data):
        """Detect N-wave pattern characteristic of sonic booms."""
        # Simplified N-wave detection
        # Look for characteristic double peak pattern
        
        # Smooth the signal
        smoothed = signal.savgol_filter(audio_data, 51, 3)
        
        # Find peaks and valleys
        peaks, _ = signal.find_peaks(smoothed, distance=len(audio_data)//10)
        valleys, _ = signal.find_peaks(-smoothed, distance=len(audio_data)//10)
        
        # N-wave has specific pattern: positive peak, negative peak, return to baseline
        if len(peaks) >= 1 and len(valleys) >= 1:
            # Check if we have the right temporal order
            first_peak = peaks[0] if len(peaks) > 0 else len(audio_data)
            first_valley = valleys[0] if len(valleys) > 0 else len(audio_data)
            
            # N-wave: positive first, then negative
            if first_peak < first_valley:
                return True
        
        return False
    
    def _detect_jet_engine_audio(self, audio_data):
        """Detect jet engine audio signature."""
        # Jet engines have characteristic broadband noise with specific frequency content
        fft_result = fft(audio_data)
        frequencies = fftfreq(len(audio_data), 1/self.sample_rate)
        magnitude = np.abs(fft_result)
        
        # Jet engines typically have:
        # - High frequency content (1-10 kHz)
        # - Broadband noise
        # - Specific harmonic structure
        
        high_freq_mask = (frequencies > 1000) & (frequencies < 10000)
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        total_energy = np.sum(magnitude)
        
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Simple jet detection heuristic
        jet_detected = high_freq_ratio > 0.3
        
        return {
            'detected': jet_detected,
            'confidence': float(min(high_freq_ratio * 2, 1.0)),
            'high_frequency_ratio': float(high_freq_ratio)
        }