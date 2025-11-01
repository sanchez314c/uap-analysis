#!/usr/bin/env python3
"""
Anomaly Detection System for Astronomical Images
=================================================
Advanced machine learning and statistical methods for detecting anomalies
in astronomical images, particularly suited for interstellar object analysis.

Features:
- Statistical anomaly detection using multiple methods
- Machine learning-based pattern recognition
- Forensic analysis for image manipulation detection
- Unusual signature detection for technological artifacts
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from scipy import stats, ndimage
from scipy.fft import fft2, fftshift
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Comprehensive anomaly detection for astronomical images"""

    def __init__(self, config=None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.one_class_svm = OneClassSVM(kernel='rbf', nu=0.1)
        self.logger = logging.getLogger(__name__)

    def analyze(self, image, reference_data=None):
        """Comprehensive anomaly detection analysis"""
        self.logger.info("Starting comprehensive anomaly detection...")

        # Extract features from image
        features = self._extract_comprehensive_features(image)

        results = {
            'statistical_anomalies': self._statistical_anomaly_detection(features, image),
            'ml_anomalies': self._ml_anomaly_detection(features),
            'pattern_anomalies': self._pattern_anomaly_detection(features),
            'frequency_anomalies': self._frequency_anomaly_detection(image),
            'geometric_anomalies': self._geometric_anomaly_detection(image),
            'forensic_analysis': self._forensic_analysis(image),
            'signature_anomalies': self._signature_anomaly_detection(image),
            'clustering_anomalies': self._clustering_anomaly_detection(features),
            'overall_anomaly_score': 0.0,
            'anomaly_summary': {}
        }

        # Calculate overall anomaly score
        results['overall_anomaly_score'] = self._calculate_overall_anomaly_score(results)
        results['anomaly_summary'] = self._generate_anomaly_summary(results)

        return results

    def _extract_comprehensive_features(self, image):
        """Extract comprehensive feature set from image"""
        features = {}

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # 1. Basic statistical features
        features['mean'] = np.mean(gray)
        features['std'] = np.std(gray)
        features['skewness'] = stats.skew(gray.flatten())
        features['kurtosis'] = stats.kurtosis(gray.flatten())
        features['min'] = np.min(gray)
        features['max'] = np.max(gray)
        features['range'] = np.max(gray) - np.min(gray)

        # 2. Texture features
        features['texture_contrast'] = self._calculate_texture_contrast(gray)
        features['texture_homogeneity'] = self._calculate_texture_homogeneity(gray)
        features['texture_entropy'] = self._calculate_texture_entropy(gray)

        # 3. Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['gradient_mean'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        features['gradient_std'] = np.std(np.sqrt(grad_x**2 + grad_y**2))

        # 4. Frequency domain features
        fft_magnitude = np.abs(fftshift(fft2(gray)))
        features['frequency_mean'] = np.mean(fft_magnitude)
        features['frequency_std'] = np.std(fft_magnitude)
        features['frequency_peak'] = np.max(fft_magnitude)

        # 5. Morphological features
        features['morphological_features'] = self._extract_morphological_features(gray)

        # 6. Color features (if color image)
        if len(image.shape) == 3:
            features['color_features'] = self._extract_color_features(image)

        return features

    def _statistical_anomaly_detection(self, features, image):
        """Statistical methods for anomaly detection"""
        anomalies = {}

        # 1. Z-score based anomaly detection
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        z_scores = np.abs(stats.zscore(gray.flatten()))
        anomalies['z_score_anomalies'] = {
            'count': np.sum(z_scores > 3),
            'percentage': float(np.sum(z_scores > 3) / len(z_scores) * 100),
            'max_z_score': float(np.max(z_scores))
        }

        # 2. Modified Z-score (more robust to outliers)
        median = np.median(gray)
        mad = np.median(np.abs(gray - median))
        modified_z_scores = 0.6745 * (gray - median) / mad
        anomalies['modified_z_score_anomalies'] = {
            'count': np.sum(np.abs(modified_z_scores) > 3.5),
            'percentage': float(np.sum(np.abs(modified_z_scores) > 3.5) / len(modified_z_scores) * 100)
        }

        # 3. IQR-based anomaly detection
        q75, q25 = np.percentile(gray, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - (1.5 * iqr)
        upper_bound = q75 + (1.5 * iqr)
        iqr_anomalies = (gray < lower_bound) | (gray > upper_bound)
        anomalies['iqr_anomalies'] = {
            'count': np.sum(iqr_anomalies),
            'percentage': float(np.sum(iqr_anomalies) / len(gray) * 100)
        }

        # 4. Chi-square goodness of fit test
        # Test if pixel distribution follows normal distribution
        _, p_value = stats.normaltest(gray.flatten())
        anomalies['distribution_test'] = {
            'is_normal': p_value > 0.05,
            'p_value': float(p_value),
            'anomaly_indicated': p_value < 0.05
        }

        return anomalies

    def _ml_anomaly_detection(self, features):
        """Machine learning based anomaly detection"""
        anomalies = {}

        # Convert features to vector format
        feature_vector = self._features_to_vector(features)

        if len(feature_vector) > 1:
            # Standardize features
            feature_vector_scaled = self.scaler.fit_transform(feature_vector.reshape(-1, 1)).flatten()

            # 1. Isolation Forest
            try:
                isolation_scores = self.isolation_forest.fit_predict(feature_vector_scaled.reshape(-1, 1))
                anomalies['isolation_forest'] = {
                    'anomaly_count': np.sum(isolation_scores == -1),
                    'anomaly_percentage': float(np.sum(isolation_scores == -1) / len(isolation_scores) * 100)
                }
            except Exception as e:
                logger.warning(f"Isolation Forest failed: {e}")
                anomalies['isolation_forest'] = {'error': str(e)}

            # 2. One-Class SVM
            try:
                svm_scores = self.one_class_svm.fit_predict(feature_vector_scaled.reshape(-1, 1))
                anomalies['one_class_svm'] = {
                    'anomaly_count': np.sum(svm_scores == -1),
                    'anomaly_percentage': float(np.sum(svm_scores == -1) / len(svm_scores) * 100)
                }
            except Exception as e:
                logger.warning(f"One-Class SVM failed: {e}")
                anomalies['one_class_svm'] = {'error': str(e)}

        return anomalies

    def _pattern_anomaly_detection(self, features):
        """Pattern-based anomaly detection"""
        anomalies = {}

        # 1. Autocorrelation analysis
        if 'morphological_features' in features:
            morph_features = features['morphological_features']
            if 'area' in morph_features:
                areas = morph_features['area'] if isinstance(morph_features['area'], list) else [morph_features['area']]
                if len(areas) > 1:
                    autocorr = np.correlate(areas, areas, mode='full')
                    expected_autocorr = np.correlate(np.ones_like(areas), np.ones_like(areas), mode='full')
                    autocorr_diff = np.abs(autocorr - expected_autocorr)
                    anomalies['autocorrelation_anomaly'] = {
                        'mean_difference': float(np.mean(autocorr_diff)),
                        'max_difference': float(np.max(autocorr_diff)),
                        'anomaly_detected': np.mean(autocorr_diff) > np.std(autocorr_diff)
                    }

        # 2. Fractal dimension analysis
        if 'texture_entropy' in features:
            entropy = features['texture_entropy']
            # Anomalous if entropy is too high or too low
            anomalies['entropy_anomaly'] = {
                'entropy_value': float(entropy),
                'anomaly_detected': entropy < 0.5 or entropy > 8.0
            }

        return anomalies

    def _frequency_anomaly_detection(self, image):
        """Frequency domain anomaly detection"""
        anomalies = {}

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # 2D FFT
        fft_result = fftshift(fft2(gray))
        magnitude = np.abs(fft_result)

        # 1. Look for unusual frequency peaks
        mean_magnitude = np.mean(magnitude)
        std_magnitude = np.std(magnitude)
        threshold = mean_magnitude + 3 * std_magnitude
        frequency_peaks = magnitude > threshold

        anomalies['frequency_peaks'] = {
            'count': np.sum(frequency_peaks),
            'percentage': float(np.sum(frequency_peaks) / frequency_peaks.size * 100),
            'peak_locations': np.argwhere(frequency_peaks).tolist()[:10]  # Top 10 locations
        }

        # 2. Radial frequency analysis
        h, w = magnitude.shape
        y, x = np.ogrid[:h, :w]
        center = (h//2, w//2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        # Calculate radial profile
        radial_bins = np.arange(0, min(h, w)//2, 5)
        radial_profile = []

        for i in range(len(radial_bins) - 1):
            mask = (r >= radial_bins[i]) & (r < radial_bins[i + 1])
            if np.any(mask):
                radial_profile.append(np.mean(magnitude[mask]))
            else:
                radial_profile.append(0)

        # Detect anomalies in radial profile
        profile_array = np.array(radial_profile)
        profile_mean = np.mean(profile_array)
        profile_std = np.std(profile_array)
        anomalies['radial_anomalies'] = {
            'anomaly_count': np.sum(np.abs(profile_array - profile_mean) > 2 * profile_std),
            'anomaly_indices': np.where(np.abs(profile_array - profile_mean) > 2 * profile_std)[0].tolist()
        }

        return anomalies

    def _geometric_anomaly_detection(self, image):
        """Geometric shape anomaly detection"""
        anomalies = {}

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Find objects
        threshold = np.percentile(gray, 99)
        binary = gray > threshold
        labeled, num_objects = ndimage.label(binary)

        geometric_features = []
        for i in range(1, min(num_objects + 1, 50)):  # Limit to 50 objects
            mask = labeled == i
            if np.sum(mask) > 5:
                # Calculate shape features
                contours, _ = cv2.findContours(mask.astype(np.uint8),
                                             cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)

                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        convex_hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(convex_hull)
                        solidity = float(area) / hull_area if hull_area > 0 else 0

                        geometric_features.append({
                            'circularity': circularity,
                            'solidity': solidity,
                            'area': area
                        })

        # Detect geometric anomalies
        if geometric_features:
            circularities = [obj['circularity'] for obj in geometric_features]
            solidities = [obj['solidity'] for obj in geometric_features]

            # Anomalous if many objects are either too circular or too irregular
            mean_circularity = np.mean(circularities)
            mean_solidity = np.mean(solidities)

            anomalies['geometric_anomalies'] = {
                'object_count': len(geometric_features),
                'mean_circularity': float(mean_circularity),
                'mean_solidity': float(mean_solidity),
                'anomalous_circularity': mean_circularity < 0.3 or mean_circularity > 0.9,
                'anomalous_solidity': mean_solidity < 0.5,
                'geometric_anomaly_detected': (mean_circularity < 0.3 or mean_circularity > 0.9) or mean_solidity < 0.5
            }

        return anomalies

    def _forensic_analysis(self, image):
        """Forensic analysis for image manipulation detection"""
        anomalies = {}

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # 1. Error Level Analysis (ELA)
        # Compress and decompress to detect manipulation traces
        compressed = cv2.imencode('.jpg', (gray * 255).astype(np.uint8))[1]
        decompressed = cv2.imdecode(compressed, cv2.IMREAD_GRAYSCALE)
        ela = np.abs(gray.astype(np.float32) - decompressed.astype(np.float32))

        ela_mean = np.mean(ela)
        ela_std = np.std(ela)
        anomalies['error_level_analysis'] = {
            'mean_ela': float(ela_mean),
            'std_ela': float(ela_std),
            'manipulation_indicated': ela_mean > 10  # Threshold for suspicious regions
        }

        # 2. Noise pattern analysis
        # Apply high-pass filter to isolate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise = cv2.filter2D(gray, cv2.CV_64F, kernel)

        # Analyze noise statistics
        noise_mean = np.mean(noise)
        noise_std = np.std(noise)

        anomalies['noise_analysis'] = {
            'noise_mean': float(noise_mean),
            'noise_std': float(noise_std),
            'inconsistent_noise': noise_std > 5  # Threshold for noise inconsistency
        }

        # 3. Clone detection
        # Look for duplicated regions (copy-paste detection)
        h, w = gray.shape
        block_size = 16
        clones_detected = 0

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]

                # Compare with other blocks
                for y2 in range(y + block_size, h - block_size, block_size):
                    for x2 in range(x + block_size, w - block_size, block_size):
                        block2 = gray[y2:y2+block_size, x2:x2+block_size]

                        # Calculate similarity
                        similarity = np.corrcoef(block.flatten(), block2.flatten())[0, 1]
                        if similarity > 0.95:  # High similarity threshold
                            clones_detected += 1

        anomalies['clone_detection'] = {
            'clones_found': clones_detected,
            'cloning_indicated': clones_detected > 5
        }

        return anomalies

    def _signature_anomaly_detection(self, image):
        """Detect unusual technological or energy signatures"""
        anomalies = {}

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        else:
            gray = image
            r = g = b = gray

        # 1. Energy emission patterns
        # Look for unusual energy distributions
        total_energy = np.sum(gray)
        channel_energies = [np.sum(r), np.sum(g), np.sum(b)]
        energy_ratios = [e / total_energy for e in channel_energies]

        anomalies['energy_signatures'] = {
            'energy_ratios': [float(r) for r in energy_ratios],
            'unusual_distribution': max(energy_ratios) > 0.7  # One channel dominates
        }

        # 2. Geometric regularity detection
        # Look for artificial geometric patterns
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            # Analyze line orientations
            angles = []
            for line in lines[:, 0]:
                x1, y1, x2, y2 = line
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

            # Check for regular angular patterns
            if angles:
                angle_hist, _ = np.histogram(angles, bins=36, range=(-180, 180))
                peak_angles = np.where(angle_hist > np.mean(angle_hist) + 2 * np.std(angle_hist))[0]

                anomalies['geometric_patterns'] = {
                    'lines_found': len(lines),
                    'regular_angles': len(peak_angles),
                    'artificial_pattern_indicated': len(peak_angles) > 2 and len(peak_angles) % 2 == 0
                }

        # 3. Symmetry detection
        # Look for unusual symmetry patterns
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])

        if left_half.shape == right_half.shape:
            symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            anomalies['symmetry_analysis'] = {
                'symmetry_score': float(symmetry_score),
                'high_symmetry_indicated': symmetry_score > 0.8
            }

        return anomalies

    def _clustering_anomaly_detection(self, features):
        """Clustering-based anomaly detection"""
        anomalies = {}

        # Convert features to clustering format
        feature_matrix = self._features_to_clustering_matrix(features)

        if feature_matrix is not None and len(feature_matrix) > 3:
            # 1. DBSCAN clustering
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=2)
                cluster_labels = dbscan.fit_predict(feature_matrix)

                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)

                anomalies['dbscan_clustering'] = {
                    'clusters_found': n_clusters,
                    'noise_points': n_noise,
                    'noise_percentage': float(n_noise / len(cluster_labels) * 100),
                    'anomaly_indicated': n_noise > len(cluster_labels) * 0.2
                }
            except Exception as e:
                logger.warning(f"DBSCAN clustering failed: {e}")
                anomalies['dbscan_clustering'] = {'error': str(e)}

            # 2. K-means clustering
            try:
                n_clusters_k = min(5, len(feature_matrix) // 2)
                kmeans = KMeans(n_clusters=n_clusters_k, random_state=42)
                kmeans_labels = kmeans.fit_predict(feature_matrix)

                # Calculate silhouette score (simplified)
                cluster_centers = kmeans.cluster_centers_
                distances = []

                for i, point in enumerate(feature_matrix):
                    center = cluster_centers[kmeans_labels[i]]
                    distances.append(np.linalg.norm(point - center))

                mean_distance = np.mean(distances)
                std_distance = np.std(distances)

                anomalies['kmeans_clustering'] = {
                    'clusters': n_clusters_k,
                    'mean_distance_to_center': float(mean_distance),
                    'distance_std': float(std_distance),
                    'anomaly_indicated': std_distance > mean_distance
                }
            except Exception as e:
                logger.warning(f"K-means clustering failed: {e}")
                anomalies['kmeans_clustering'] = {'error': str(e)}

        return anomalies

    def _features_to_vector(self, features):
        """Convert features dictionary to numerical vector"""
        vector = []

        # Add basic features
        basic_features = ['mean', 'std', 'skewness', 'kurtosis', 'min', 'max', 'range']
        for feature in basic_features:
            if feature in features:
                vector.append(features[feature])

        # Add texture features
        texture_features = ['texture_contrast', 'texture_homogeneity', 'texture_entropy']
        for feature in texture_features:
            if feature in features:
                vector.append(features[feature])

        # Add gradient features
        gradient_features = ['gradient_mean', 'gradient_std']
        for feature in gradient_features:
            if feature in features:
                vector.append(features[feature])

        # Add frequency features
        frequency_features = ['frequency_mean', 'frequency_std', 'frequency_peak']
        for feature in frequency_features:
            if feature in features:
                vector.append(features[feature])

        return np.array(vector)

    def _features_to_clustering_matrix(self, features):
        """Convert features to matrix suitable for clustering"""
        # Create feature vectors for different regions or objects
        matrix = []

        if 'morphological_features' in features:
            morph = features['morphological_features']
            if isinstance(morph, dict):
                # If we have multiple objects, create feature vectors
                for key, value in morph.items():
                    if isinstance(value, list) and len(value) > 1:
                        for v in value:
                            matrix.append([v])
                    else:
                        matrix.append([float(value)])

        if not matrix and len(self._features_to_vector(features)) > 0:
            # If no object-based features, use single image features
            vector = self._features_to_vector(features)
            matrix = [vector[i:i+3] for i in range(0, len(vector), 3)]

        return np.array(matrix) if matrix else None

    def _calculate_texture_contrast(self, gray):
        """Calculate texture contrast using GLCM approximation"""
        # Simple texture contrast calculation
        h, w = gray.shape
        contrast = 0
        count = 0

        for i in range(0, h-1, 10):  # Sample every 10 pixels for speed
            for j in range(0, w-1, 10):
                diff = abs(gray[i, j] - gray[i+1, j+1])
                contrast += diff
                count += 1

        return contrast / count if count > 0 else 0

    def _calculate_texture_homogeneity(self, gray):
        """Calculate texture homogeneity"""
        # Simple homogeneity calculation
        h, w = gray.shape
        homogeneity = 0
        count = 0

        for i in range(0, h-1, 10):
            for j in range(0, w-1, 10):
                diff = abs(gray[i, j] - gray[i+1, j+1])
                homogeneity += 1 / (1 + diff)
                count += 1

        return homogeneity / count if count > 0 else 0

    def _calculate_texture_entropy(self, gray):
        """Calculate texture entropy"""
        # Calculate histogram
        hist, _ = np.histogram(gray.flatten(), bins=256, density=True)
        hist = hist[hist > 0]  # Remove zero entries

        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy

    def _extract_morphological_features(self, gray):
        """Extract morphological features from image"""
        # Find objects
        threshold = np.percentile(gray, 99)
        binary = gray > threshold
        labeled, num_objects = ndimage.label(binary)

        features = {}
        if num_objects > 0:
            # Calculate properties for all objects
            areas = []
            for i in range(1, num_objects + 1):
                area = np.sum(labeled == i)
                if area > 5:  # Minimum size threshold
                    areas.append(area)

            if areas:
                features['area'] = areas
                features['num_objects'] = len(areas)
                features['mean_area'] = np.mean(areas)
                features['std_area'] = np.std(areas)

        return features

    def _extract_color_features(self, image):
        """Extract color features from image"""
        features = {}

        # Calculate color statistics
        for i, channel in enumerate(['r', 'g', 'b']):
            channel_data = image[:, :, i]
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)

        # Calculate color ratios
        r_mean = features['r_mean']
        g_mean = features['g_mean']
        b_mean = features['b_mean']

        features['r_g_ratio'] = r_mean / (g_mean + 1e-10)
        features['g_b_ratio'] = g_mean / (b_mean + 1e-10)
        features['r_b_ratio'] = r_mean / (b_mean + 1e-10)

        return features

    def _calculate_overall_anomaly_score(self, results):
        """Calculate overall anomaly score from all detection methods"""
        score = 0.0
        weights = {
            'statistical_anomalies': 0.2,
            'ml_anomalies': 0.15,
            'pattern_anomalies': 0.1,
            'frequency_anomalies': 0.15,
            'geometric_anomalies': 0.1,
            'forensic_analysis': 0.15,
            'signature_anomalies': 0.1,
            'clustering_anomalies': 0.05
        }

        for method, weight in weights.items():
            if method in results:
                method_score = self._calculate_method_score(results[method])
                score += weight * method_score

        return float(np.clip(score, 0.0, 1.0))

    def _calculate_method_score(self, method_results):
        """Calculate anomaly score for a specific detection method"""
        if isinstance(method_results, dict):
            if 'error' in method_results:
                return 0.0

            # Look for boolean anomaly indicators
            for key, value in method_results.items():
                if isinstance(value, bool) and value:
                    return 1.0
                elif isinstance(value, (int, float)) and value > 0:
                    # Normalize numeric values
                    if 'percentage' in key or 'count' in key:
                        return min(1.0, value / 10.0) if isinstance(value, (int, float)) else 0.0

        return 0.0

    def _generate_anomaly_summary(self, results):
        """Generate human-readable summary of anomalies"""
        summary = {
            'total_anomaly_types': 0,
            'critical_anomalies': [],
            'warnings': [],
            'recommendations': []
        }

        for method, result in results.items():
            if method.endswith('anomalies') and isinstance(result, dict):
                if 'error' not in result:
                    summary['total_anomaly_types'] += 1

                    # Check for critical anomalies
                    for key, value in result.items():
                        if isinstance(value, bool) and value:
                            summary['critical_anomalies'].append(f"{method}: {key}")
                        elif isinstance(value, (int, float)) and value > 5:
                            summary['warnings'].append(f"{method}: {key} = {value}")

        # Generate recommendations
        if summary['total_anomaly_types'] > 5:
            summary['recommendations'].append("High number of anomaly types detected - recommend expert review")

        if any('forensic' in anomaly for anomaly in summary['critical_anomalies']):
            summary['recommendations'].append("Potential image manipulation detected - verify source authenticity")

        if any('signature' in anomaly for anomaly in summary['critical_anomalies']):
            summary['recommendations'].append("Unusual signatures detected - may indicate technological origin")

        return summary