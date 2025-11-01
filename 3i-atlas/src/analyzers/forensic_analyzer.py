#!/usr/bin/env python3
"""
Forensic Image Analyzer for Astronomical Images
===============================================
Specialized forensic analysis tools for astronomical images,
focusing on detecting manipulation, artifacts, and authenticity verification.

Features:
- Image manipulation detection using multiple algorithms
- Compression artifact analysis
- Metadata extraction and validation
- Sensor pattern analysis
- Cosmic ray and radiation artifact detection
- Digital watermark and steganography detection
"""

import cv2
import numpy as np
from scipy import fft, ndimage, stats
from scipy.signal import convolve2d
import hashlib
import struct
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ForensicAnalyzer:
    """Comprehensive forensic analysis for astronomical images"""

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def analyze(self, image_path, image_data=None):
        """Comprehensive forensic analysis"""
        self.logger.info("Starting forensic analysis...")

        # Load image if not provided
        if image_data is None:
            image_data = self._load_image(image_path)

        results = {
            'metadata_analysis': self._analyze_metadata(image_path),
            'manipulation_detection': self._detect_manipulation(image_data),
            'compression_analysis': self._analyze_compression(image_data, image_path),
            'sensor_pattern_analysis': self._analyze_sensor_patterns(image_data),
            'cosmic_ray_detection': self._detect_cosmic_rays(image_data),
            'radiation_artifacts': self._detect_radiation_artifacts(image_data),
            'authenticity_score': 0.0,
            'forensic_summary': {}
        }

        # Calculate overall authenticity score
        results['authenticity_score'] = self._calculate_authenticity_score(results)
        results['forensic_summary'] = self._generate_forensic_summary(results)

        return results

    def _load_image(self, image_path):
        """Load image with error handling"""
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Convert to float64 for analysis
            if image.dtype == np.uint16:
                image = image.astype(np.float64) / 65535.0
            elif image.dtype == np.uint8:
                image = image.astype(np.float64) / 255.0
            else:
                image = image.astype(np.float64)

            return image

        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            raise

    def _analyze_metadata(self, image_path):
        """Analyze image metadata"""
        metadata = {
            'file_info': {},
            'exif_data': {},
            'checksums': {},
            'suspicious_indicators': []
        }

        try:
            # Basic file information
            file_path = Path(image_path)
            file_stat = file_path.stat()

            metadata['file_info'] = {
                'filename': file_path.name,
                'size_bytes': file_stat.st_size,
                'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'extension': file_path.suffix.lower()
            }

            # Calculate file checksums
            with open(image_path, 'rb') as f:
                file_content = f.read()

            metadata['checksums'] = {
                'md5': hashlib.md5(file_content).hexdigest(),
                'sha256': hashlib.sha256(file_content).hexdigest()
            }

            # Check for suspicious indicators
            if metadata['file_info']['size_bytes'] < 1000:
                metadata['suspicious_indicators'].append("Very small file size")

            if metadata['file_info']['extension'] not in ['.tif', '.tiff', '.fits', '.png']:
                metadata['suspicious_indicators'].append("Unusual file extension for astronomical data")

            # Time analysis
            creation_time = file_stat.st_ctime
            modification_time = file_stat.st_mtime
            if abs(creation_time - modification_time) < 1:
                metadata['suspicious_indicators'].append("File created and modified almost simultaneously")

        except Exception as e:
            self.logger.warning(f"Metadata analysis failed: {e}")
            metadata['error'] = str(e)

        return metadata

    def _detect_manipulation(self, image):
        """Detect various forms of image manipulation"""
        manipulation = {
            'error_level_analysis': {},
            'noise_inconsistency': {},
            'lighting_inconsistency': {},
            'cloning_detection': {},
            'resampling_detection': {},
            'manipulation_probability': 0.0
        }

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        try:
            # 1. Error Level Analysis (ELA)
            manipulation['error_level_analysis'] = self._perform_ela(gray)

            # 2. Noise inconsistency analysis
            manipulation['noise_inconsistency'] = self._analyze_noise_inconsistency(gray)

            # 3. Lighting inconsistency analysis
            manipulation['lighting_inconsistency'] = self._analyze_lighting_inconsistency(image)

            # 4. Cloning detection
            manipulation['cloning_detection'] = self._detect_cloning(gray)

            # 5. Resampling detection
            manipulation['resampling_detection'] = self._detect_resampling(gray)

            # Calculate overall manipulation probability
            manipulation['manipulation_probability'] = self._calculate_manipulation_probability(manipulation)

        except Exception as e:
            self.logger.warning(f"Manipulation detection failed: {e}")
            manipulation['error'] = str(e)

        return manipulation

    def _perform_ela(self, gray):
        """Perform Error Level Analysis"""
        # Save image at specific quality level
        quality = 75
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', (gray * 255).astype(np.uint8), encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        # Calculate difference
        ela = np.abs(gray.astype(np.float32) - decoded.astype(np.float32))

        # Analyze ELA results
        ela_mean = np.mean(ela)
        ela_std = np.std(ela)
        ela_max = np.max(ela)

        # Look for suspicious patterns
        high_ela_pixels = np.sum(ela > ela_mean + 2 * ela_std)
        total_pixels = ela.size
        high_ela_percentage = (high_ela_pixels / total_pixels) * 100

        return {
            'mean_ela': float(ela_mean),
            'std_ela': float(ela_std),
            'max_ela': float(ela_max),
            'high_ela_percentage': float(high_ela_percentage),
            'manipulation_indicated': high_ela_percentage > 5.0 or ela_max > 30
        }

    def _analyze_noise_inconsistency(self, gray):
        """Analyze noise patterns for inconsistencies"""
        # Divide image into blocks and analyze noise in each
        h, w = gray.shape
        block_size = 64
        noise_levels = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]

                # Estimate noise using high-pass filter
                kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                noise = convolve2d(block, kernel, mode='same')
                noise_std = np.std(noise)
                noise_levels.append(noise_std)

        if noise_levels:
            noise_mean = np.mean(noise_levels)
            noise_std = np.std(noise_levels)
            noise_cv = noise_std / (noise_mean + 1e-10)  # Coefficient of variation

            # Look for outliers
            outliers = [n for n in noise_levels if abs(n - noise_mean) > 2 * noise_std]
            outlier_percentage = (len(outliers) / len(noise_levels)) * 100

            return {
                'mean_noise': float(noise_mean),
                'noise_std': float(noise_std),
                'noise_cv': float(noise_cv),
                'outlier_percentage': float(outlier_percentage),
                'inconsistency_indicated': outlier_percentage > 15 or noise_cv > 0.5
            }

        return {'error': 'Insufficient data for noise analysis'}

    def _analyze_lighting_inconsistency(self, image):
        """Analyze lighting consistency across the image"""
        if len(image.shape) != 3:
            return {'error': 'Color image required for lighting analysis'}

        # Convert to uint8 for OpenCV processing if needed
        if image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        # Convert to different color spaces
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)

        # Analyze lighting direction and consistency
        light_directions = []

        # Sample patches from different regions
        h, w = image.shape[:2]
        patch_size = 32

        for y in range(0, h - patch_size, patch_size * 2):
            for x in range(0, w - patch_size, patch_size * 2):
                patch = image[y:y+patch_size, x:x+patch_size]

                # Estimate lighting direction from gradients
                gray_patch = np.mean(patch, axis=2)
                grad_x = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)

                # Calculate dominant gradient direction
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                if np.sum(grad_magnitude) > 0:
                    avg_grad_x = np.sum(grad_x) / np.sum(grad_magnitude)
                    avg_grad_y = np.sum(grad_y) / np.sum(grad_magnitude)
                    direction = np.arctan2(avg_grad_y, avg_grad_x)
                    light_directions.append(direction)

        if len(light_directions) > 1:
            # Calculate direction consistency
            direction_std = np.std(light_directions)
            direction_range = np.max(light_directions) - np.min(light_directions)

            return {
                'direction_consistency': float(1 / (1 + direction_std)),
                'direction_range': float(direction_range),
                'inconsistency_indicated': direction_std > 0.5 or direction_range > np.pi
            }

        return {'error': 'Insufficient data for lighting analysis'}

    def _detect_cloning(self, gray):
        """Detect cloned or copy-pasted regions"""
        h, w = gray.shape
        block_size = 16
        clones = []

        # Extract all possible blocks
        blocks = {}
        for y in range(0, h - block_size, 4):  # Overlap blocks
            for x in range(0, w - block_size, 4):
                block = gray[y:y+block_size, x:x+block_size]
                block_hash = hash(block.tobytes())

                if block_hash in blocks:
                    blocks[block_hash].append((x, y))
                else:
                    blocks[block_hash] = [(x, y)]

        # Look for duplicate blocks
        for block_hash, positions in blocks.items():
            if len(positions) > 1:
                # Check similarity more carefully
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        x1, y1 = positions[i]
                        x2, y2 = positions[j]

                        # Ensure blocks are not overlapping
                        if (abs(x1 - x2) > block_size or abs(y1 - y2) > block_size):
                            block1 = gray[y1:y1+block_size, x1:x1+block_size]
                            block2 = gray[y2:y2+block_size, x2:x2+block_size]

                            # Calculate correlation
                            correlation = np.corrcoef(block1.flatten(), block2.flatten())[0, 1]
                            if correlation > 0.95:
                                clones.append({
                                    'region1': (x1, y1, block_size),
                                    'region2': (x2, y2, block_size),
                                    'correlation': float(correlation)
                                })

        return {
            'clones_found': len(clones),
            'clone_details': clones[:5],  # Return first 5 clones
            'cloning_indicated': len(clones) > 3
        }

    def _detect_resampling(self, gray):
        """Detect resampling artifacts"""
        # Analyze periodic patterns in pixel differences
        diff_x = np.diff(gray, axis=1)
        diff_y = np.diff(gray, axis=0)

        # FFT of differences
        fft_diff_x = np.abs(fft.fft2(diff_x))
        fft_diff_y = np.abs(fft.fft2(diff_y))

        # Look for peaks in frequency domain (indicative of resampling)
        mean_fft_x = np.mean(fft_diff_x)
        std_fft_x = np.std(fft_diff_x)
        peaks_x = np.sum(fft_diff_x > mean_fft_x + 3 * std_fft_x)

        mean_fft_y = np.mean(fft_diff_y)
        std_fft_y = np.std(fft_diff_y)
        peaks_y = np.sum(fft_diff_y > mean_fft_y + 3 * std_fft_y)

        return {
            'horizontal_peaks': int(peaks_x),
            'vertical_peaks': int(peaks_y),
            'total_peaks': int(peaks_x + peaks_y),
            'resampling_indicated': (peaks_x + peaks_y) > 100
        }

    def _analyze_compression(self, image, image_path):
        """Analyze compression artifacts and history"""
        compression = {
            'jpeg_artifacts': {},
            'compression_history': {},
            'quantization_analysis': {},
            'dct_coefficients': {},
            'compression_detected': False
        }

        try:
            # Check for JPEG artifacts
            compression['jpeg_artifacts'] = self._detect_jpeg_artifacts(image)

            # Analyze DCT coefficients (if JPEG)
            if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                compression['dct_coefficients'] = self._analyze_dct_coefficients(image)

            # Check for multiple compression generations
            compression['compression_history'] = self._detect_compression_history(image)

            # Overall compression detection
            compression['compression_detected'] = (
                compression['jpeg_artifacts'].get('artifacts_detected', False) or
                compression['compression_history'].get('multiple_compressions', False)
            )

        except Exception as e:
            self.logger.warning(f"Compression analysis failed: {e}")
            compression['error'] = str(e)

        return compression

    def _detect_jpeg_artifacts(self, image):
        """Detect JPEG compression artifacts"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Look for 8x8 block artifacts (JPEG characteristic)
        h, w = gray.shape
        block_size = 8

        # Calculate differences at block boundaries
        horizontal_artifacts = 0
        vertical_artifacts = 0

        for y in range(block_size, h, block_size):
            diff = np.abs(gray[y, :] - gray[y-1, :])
            horizontal_artifacts += np.sum(diff > np.mean(diff) + 2 * np.std(diff))

        for x in range(block_size, w, block_size):
            diff = np.abs(gray[:, x] - gray[:, x-1])
            vertical_artifacts += np.sum(diff > np.mean(diff) + 2 * np.std(diff))

        total_artifacts = horizontal_artifacts + vertical_artifacts
        artifact_density = total_artifacts / (h + w)

        return {
            'horizontal_artifacts': int(horizontal_artifacts),
            'vertical_artifacts': int(vertical_artifacts),
            'total_artifacts': int(total_artifacts),
            'artifact_density': float(artifact_density),
            'artifacts_detected': artifact_density > 0.1
        }

    def _analyze_dct_coefficients(self, image):
        """Analyze DCT coefficients for JPEG images"""
        # This is a simplified analysis - real DCT analysis would be more complex
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Apply DCT to 8x8 blocks and analyze coefficient distribution
        h, w = gray.shape
        block_size = 8
        coefficients = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                dct_block = cv2.dct(block.astype(np.float32))
                coefficients.extend(dct_block.flatten())

        if coefficients:
            coeff_array = np.array(coefficients)
            coeff_mean = np.mean(coeff_array)
            coeff_std = np.std(coeff_array)
            zero_coeff_ratio = np.sum(coeff_array == 0) / len(coeff_array)

            return {
                'coefficient_mean': float(coeff_mean),
                'coefficient_std': float(coeff_std),
                'zero_coefficient_ratio': float(zero_coeff_ratio),
                'quantization_indicated': zero_coeff_ratio > 0.3
            }

        return {'error': 'Insufficient data for DCT analysis'}

    def _detect_compression_history(self, image):
        """Detect signs of multiple compression generations"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Look for double compression artifacts
        # This is a simplified approach

        # Apply a compression-decompression cycle
        compressed = cv2.imencode('.jpg', (gray * 255).astype(np.uint8))[1]
        decompressed = cv2.imdecode(compressed, cv2.IMREAD_GRAYSCALE)

        # Compare with original
        diff = np.abs(gray.astype(np.float32) - decompressed.astype(np.float32))
        diff_mean = np.mean(diff)
        diff_std = np.std(diff)

        # High variation might indicate multiple compressions
        variation_blocks = []
        block_size = 32
        h, w = gray.shape

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block_diff = diff[y:y+block_size, x:x+block_size]
                variation_blocks.append(np.std(block_diff))

        if variation_blocks:
            variation_std = np.std(variation_blocks)
            multiple_compression_indicated = variation_std > diff_std * 2

            return {
                'compression_difference_mean': float(diff_mean),
                'compression_difference_std': float(diff_std),
                'block_variation_std': float(variation_std),
                'multiple_compressions': multiple_compression_indicated
            }

        return {'error': 'Insufficient data for compression history analysis'}

    def _analyze_sensor_patterns(self, image):
        """Analyze sensor patterns and defects"""
        patterns = {
            'hot_pixels': {},
            'dead_pixels': {},
            'pattern_noise': {},
            'sensor_dust': {},
            'cosmic_ray_hits': {}
        }

        try:
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image

            # Detect hot pixels
            patterns['hot_pixels'] = self._detect_hot_pixels(gray)

            # Detect dead pixels
            patterns['dead_pixels'] = self._detect_dead_pixels(gray)

            # Analyze pattern noise
            patterns['pattern_noise'] = self._analyze_pattern_noise(gray)

            # Detect sensor dust
            patterns['sensor_dust'] = self._detect_sensor_dust(gray)

            # Detect cosmic ray hits
            patterns['cosmic_ray_hits'] = self._detect_cosmic_ray_hits(gray)

        except Exception as e:
            self.logger.warning(f"Sensor pattern analysis failed: {e}")
            patterns['error'] = str(e)

        return patterns

    def _detect_hot_pixels(self, gray):
        """Detect hot pixels (abnormally bright pixels)"""
        threshold = np.percentile(gray, 99.9)
        hot_pixels = np.where(gray > threshold)

        return {
            'count': len(hot_pixels[0]),
            'locations': list(zip(hot_pixels[1], hot_pixels[0]))[:10],  # First 10 locations
            'brightness_threshold': float(threshold)
        }

    def _detect_dead_pixels(self, gray):
        """Detect dead pixels (abnormally dark pixels)"""
        threshold = np.percentile(gray, 0.1)
        dead_pixels = np.where(gray < threshold)

        return {
            'count': len(dead_pixels[0]),
            'locations': list(zip(dead_pixels[1], dead_pixels[0]))[:10],
            'darkness_threshold': float(threshold)
        }

    def _analyze_pattern_noise(self, gray):
        """Analyze fixed pattern noise"""
        # Extract high-frequency noise
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        noise = convolve2d(gray, kernel, mode='same')

        # Look for repeating patterns
        h, w = noise.shape
        sample_size = min(100, h//4, w//4)

        # Sample different regions
        samples = []
        for i in range(4):
            y = i * (h // 4)
            x = i * (w // 4)
            sample = noise[y:y+sample_size, x:x+sample_size]
            samples.append(sample)

        # Calculate correlation between samples
        correlations = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                corr = np.corrcoef(samples[i].flatten(), samples[j].flatten())[0, 1]
                correlations.append(corr)

        if correlations:
            mean_correlation = np.mean(correlations)
            pattern_noise_detected = mean_correlation > 0.3

            return {
                'mean_correlation': float(mean_correlation),
                'pattern_noise_detected': pattern_noise_detected,
                'correlation_samples': len(correlations)
            }

        return {'error': 'Insufficient data for pattern noise analysis'}

    def _detect_sensor_dust(self, gray):
        """Detect sensor dust spots"""
        # Dust spots appear as dark, soft-edged circles
        threshold = np.percentile(gray, 5)
        dark_areas = gray < threshold

        # Find connected dark regions
        labeled, num_features = ndimage.label(dark_areas)

        dust_spots = []
        for i in range(1, min(num_features + 1, 20)):  # Limit to 20 features
            region = labeled == i
            area = np.sum(region)

            if 10 < area < 1000:  # Dust spot size range
                # Check circularity
                y_coords, x_coords = np.where(region)
                if len(x_coords) > 0:
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                    radius_std = np.std(distances) / (np.mean(distances) + 1e-10)

                    if radius_std < 0.3:  # Reasonably circular
                        dust_spots.append({
                            'center': (float(center_x), float(center_y)),
                            'area': int(area),
                            'radius_estimate': float(np.mean(distances))
                        })

        return {
            'dust_spots_found': len(dust_spots),
            'dust_spot_details': dust_spots
        }

    def _detect_cosmic_ray_hits(self, gray):
        """Detect cosmic ray hits in the image"""
        # Cosmic rays appear as bright, linear or pixel-wide features
        threshold = np.percentile(gray, 99.95)
        bright_pixels = gray > threshold

        # Use morphological operations to find linear features
        kernel_line = np.ones((1, 5))  # Horizontal line kernel
        kernel_line_v = np.ones((5, 1))  # Vertical line kernel

        horizontal_hits = ndimage.convolve(bright_pixels.astype(float), kernel_line)
        vertical_hits = ndimage.convolve(bright_pixels.astype(float), kernel_line_v)

        # Combine detections
        ray_hits = np.logical_or(horizontal_hits > 2, vertical_hits > 2)

        # Count and analyze hits
        labeled, num_hits = ndimage.label(ray_hits)

        hits_info = []
        for i in range(1, min(num_hits + 1, 10)):  # Limit to 10 hits
            hit_region = labeled == i
            y_coords, x_coords = np.where(hit_region)

            if len(x_coords) > 1:
                # Calculate hit properties
                length = max(np.max(x_coords) - np.min(x_coords),
                           np.max(y_coords) - np.min(y_coords))
                intensity = np.mean(gray[hit_region])

                hits_info.append({
                    'length': int(length),
                    'mean_intensity': float(intensity),
                    'pixel_count': int(len(x_coords))
                })

        return {
            'cosmic_ray_hits': num_hits,
            'hit_details': hits_info,
            'high_energy_events': len([h for h in hits_info if h['mean_intensity'] > threshold * 1.1])
        }

    def _detect_cosmic_rays(self, image):
        """Specialized cosmic ray detection (alias for _detect_cosmic_ray_hits)"""
        return self._detect_cosmic_ray_hits(image)

    def _detect_radiation_artifacts(self, image):
        """Detect radiation-related artifacts"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Look for various radiation artifacts
        artifacts = {
            'pixel_clusters': {},
            'stuck_pixels': {},
            'transient_events': {}
        }

        # Detect abnormal pixel clusters
        threshold_high = np.percentile(gray, 99.9)
        threshold_low = np.percentile(gray, 0.1)

        high_clusters = gray > threshold_high
        low_clusters = gray < threshold_low

        # Analyze clusters
        artifacts['pixel_clusters'] = {
            'high_intensity_clusters': int(ndimage.label(high_clusters)[1]),
            'low_intensity_clusters': int(ndimage.label(low_clusters)[1])
        }

        # Detect stuck pixels (consistently high or low)
        # This would require multiple exposures, so we'll do a basic check
        stuck_high = np.sum(gray > threshold_high * 1.2)
        stuck_low = np.sum(gray < threshold_low * 0.8)

        artifacts['stuck_pixels'] = {
            'stuck_high_count': int(stuck_high),
            'stuck_low_count': int(stuck_low),
            'total_stuck_pixels': int(stuck_high + stuck_low)
        }

        return artifacts

    def _calculate_manipulation_probability(self, manipulation_results):
        """Calculate overall manipulation probability"""
        probability = 0.0
        evidence_count = 0

        for method, result in manipulation_results.items():
            if isinstance(result, dict) and 'error' not in result:
                # Look for manipulation indicators
                for key, value in result.items():
                    if 'indicated' in key and value:
                        probability += 0.2
                        evidence_count += 1
                    elif 'probability' in key:
                        probability += value * 0.3
                        evidence_count += 1

        # Normalize probability
        if evidence_count > 0:
            probability = min(1.0, probability / evidence_count)

        return float(probability)

    def _calculate_authenticity_score(self, results):
        """Calculate overall authenticity score"""
        score = 1.0  # Start with perfect authenticity

        # Deduct points for manipulation evidence
        if 'manipulation_detection' in results:
            manipulation_prob = results['manipulation_detection'].get('manipulation_probability', 0)
            score -= manipulation_prob * 0.4

        # Deduct points for suspicious metadata
        if 'metadata_analysis' in results:
            metadata = results['metadata_analysis']
            if 'suspicious_indicators' in metadata:
                score -= len(metadata['suspicious_indicators']) * 0.1

        # Deduct points for compression artifacts (if unexpected)
        if 'compression_analysis' in results:
            compression = results['compression_analysis']
            if compression.get('compression_detected', False):
                score -= 0.1

        # Account for cosmic ray hits (these are expected in astronomical images)
        if 'sensor_pattern_analysis' in results:
            sensor = results['sensor_pattern_analysis']
            if 'cosmic_ray_hits' in sensor:
                cosmic_hits = sensor['cosmic_ray_hits'].get('cosmic_ray_hits', 0)
                # Cosmic rays actually increase authenticity for astronomical images
                if cosmic_hits > 0:
                    score = min(1.0, score + 0.1)

        return max(0.0, min(1.0, score))

    def _generate_forensic_summary(self, results):
        """Generate human-readable forensic summary"""
        summary = {
            'authenticity_assessment': 'AUTHENTIC',
            'confidence_level': 'HIGH',
            'key_findings': [],
            'recommendations': [],
            'risk_level': 'LOW'
        }

        authenticity_score = results.get('authenticity_score', 0.5)

        # Determine authenticity assessment
        if authenticity_score > 0.8:
            summary['authenticity_assessment'] = 'AUTHENTIC'
            summary['confidence_level'] = 'HIGH'
            summary['risk_level'] = 'LOW'
        elif authenticity_score > 0.6:
            summary['authenticity_assessment'] = 'LIKELY AUTHENTIC'
            summary['confidence_level'] = 'MEDIUM'
            summary['risk_level'] = 'MEDIUM'
        elif authenticity_score > 0.4:
            summary['authenticity_assessment'] = 'UNCERTAIN'
            summary['confidence_level'] = 'LOW'
            summary['risk_level'] = 'HIGH'
        else:
            summary['authenticity_assessment'] = 'LIKELY MANIPULATED'
            summary['confidence_level'] = 'HIGH'
            summary['risk_level'] = 'CRITICAL'

        # Add key findings
        if 'manipulation_detection' in results:
            manipulation = results['manipulation_detection']
            if manipulation.get('manipulation_probability', 0) > 0.5:
                summary['key_findings'].append("Significant manipulation evidence detected")

        if 'sensor_pattern_analysis' in results:
            sensor = results['sensor_pattern_analysis']
            if 'cosmic_ray_hits' in sensor:
                hits = sensor['cosmic_ray_hits'].get('cosmic_ray_hits', 0)
                if hits > 0:
                    summary['key_findings'].append(f"Detected {hits} cosmic ray hits (consistent with astronomical exposure)")

        # Generate recommendations
        if authenticity_score < 0.6:
            summary['recommendations'].append("Verify image source and acquisition method")
            summary['recommendations'].append("Cross-reference with other observations if available")
            summary['recommendations'].append("Consider expert forensic analysis")

        if summary['risk_level'] in ['HIGH', 'CRITICAL']:
            summary['recommendations'].append("Exercise caution when using this image for scientific analysis")

        return summary