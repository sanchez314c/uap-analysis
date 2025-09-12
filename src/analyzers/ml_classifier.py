#!/usr/bin/env python3
"""
Machine Learning Classification Component
========================================

Uses machine learning to classify and identify patterns in UAP footage,
including anomaly detection and pattern recognition.
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class MLClassifier:
    """Machine learning-based pattern classification and anomaly detection."""
    
    def __init__(self, config):
        """Initialize ML classifier."""
        self.config = config
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def analyze(self, frames, metadata):
        """Perform ML-based analysis on video frames."""
        logger.info("Starting ML classification analysis...")
        
        # Extract features from all frames
        features = self._extract_features(frames)
        
        results = {
            'feature_analysis': self._analyze_features(features),
            'anomaly_detection': self._detect_anomalies(features),
            'pattern_clustering': self._cluster_patterns(features),
            'object_classification': self._classify_objects(frames),
            'behavioral_analysis': self._analyze_behavior_patterns(features, metadata),
            'similarity_analysis': self._analyze_frame_similarity(features)
        }
        
        return results
    
    def _extract_features(self, frames):
        """Extract comprehensive features from video frames."""
        logger.info("Extracting features from frames...")
        
        features = []
        
        for i, frame in enumerate(frames):
            frame_features = {}
            
            # Basic image statistics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_features.update(self._extract_basic_features(gray))
            
            # Texture features
            frame_features.update(self._extract_texture_features(gray))
            
            # Color features
            frame_features.update(self._extract_color_features(frame))
            
            # Shape features
            frame_features.update(self._extract_shape_features(gray))
            
            # Frequency domain features
            frame_features.update(self._extract_frequency_features(gray))
            
            # Motion features (if not first frame)
            if i > 0:
                prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                frame_features.update(self._extract_motion_features(prev_gray, gray))
            else:
                # Fill with zeros for first frame
                frame_features.update(self._get_default_motion_features())
            
            features.append(frame_features)
        
        return features
    
    def _extract_basic_features(self, gray_image):
        """Extract basic statistical features."""
        return {
            'mean_intensity': float(np.mean(gray_image)),
            'std_intensity': float(np.std(gray_image)),
            'min_intensity': float(np.min(gray_image)),
            'max_intensity': float(np.max(gray_image)),
            'intensity_range': float(np.max(gray_image) - np.min(gray_image)),
            'skewness': float(self._calculate_skewness(gray_image)),
            'kurtosis': float(self._calculate_kurtosis(gray_image))
        }
    
    def _extract_texture_features(self, gray_image):
        """Extract texture-based features using various methods."""
        # Local Binary Pattern (simplified)
        lbp = self._calculate_lbp(gray_image)
        
        # Gradient features
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge density
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'lbp_mean': float(np.mean(lbp)),
            'lbp_std': float(np.std(lbp)),
            'gradient_mean': float(np.mean(gradient_magnitude)),
            'gradient_std': float(np.std(gradient_magnitude)),
            'edge_density': float(edge_density),
            'texture_contrast': float(np.std(gradient_magnitude))
        }
    
    def _extract_color_features(self, color_image):
        """Extract color-based features."""
        # Convert to different color spaces
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
        
        # Color histograms
        hist_b = cv2.calcHist([color_image], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([color_image], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([color_image], [2], None, [32], [0, 256])
        
        # Color moments
        b, g, r = cv2.split(color_image)
        
        return {
            'color_mean_b': float(np.mean(b)),
            'color_mean_g': float(np.mean(g)),
            'color_mean_r': float(np.mean(r)),
            'color_std_b': float(np.std(b)),
            'color_std_g': float(np.std(g)),
            'color_std_r': float(np.std(r)),
            'hue_mean': float(np.mean(hsv[:,:,0])),
            'saturation_mean': float(np.mean(hsv[:,:,1])),
            'value_mean': float(np.mean(hsv[:,:,2])),
            'color_entropy': float(self._calculate_color_entropy(color_image))
        }
    
    def _extract_shape_features(self, gray_image):
        """Extract shape-based features."""
        # Find contours
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._get_default_shape_features()
        
        # Analyze largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Shape properties
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        return {
            'object_area': float(area),
            'object_perimeter': float(perimeter),
            'aspect_ratio': float(aspect_ratio),
            'solidity': float(solidity),
            'circularity': float(circularity),
            'bounding_width': float(w),
            'bounding_height': float(h),
            'contour_count': len(contours)
        }
    
    def _extract_frequency_features(self, gray_image):
        """Extract frequency domain features."""
        # FFT analysis
        f_transform = np.fft.fft2(gray_image)
        magnitude_spectrum = np.abs(f_transform)
        
        # Energy in different frequency bands
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency energy (center region)
        low_freq_region = magnitude_spectrum[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8]
        low_freq_energy = np.sum(low_freq_region)
        
        # High frequency energy
        high_freq_mask = np.ones_like(magnitude_spectrum)
        high_freq_mask[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8] = 0
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
        
        total_energy = np.sum(magnitude_spectrum)
        
        return {
            'low_freq_energy': float(low_freq_energy),
            'high_freq_energy': float(high_freq_energy),
            'freq_energy_ratio': float(low_freq_energy / total_energy) if total_energy > 0 else 0,
            'spectral_centroid': float(self._calculate_spectral_centroid(magnitude_spectrum))
        }
    
    def _extract_motion_features(self, prev_frame, curr_frame):
        """Extract motion-based features."""
        # Optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Flow magnitude and direction
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Frame difference
        frame_diff = cv2.absdiff(prev_frame, curr_frame)
        
        return {
            'motion_magnitude_mean': float(np.mean(magnitude)),
            'motion_magnitude_std': float(np.std(magnitude)),
            'motion_magnitude_max': float(np.max(magnitude)),
            'motion_angle_std': float(np.std(angle)),
            'frame_diff_mean': float(np.mean(frame_diff)),
            'frame_diff_std': float(np.std(frame_diff)),
            'motion_density': float(np.sum(magnitude > 1.0) / magnitude.size)
        }
    
    def _get_default_motion_features(self):
        """Default motion features for first frame."""
        return {
            'motion_magnitude_mean': 0.0,
            'motion_magnitude_std': 0.0,
            'motion_magnitude_max': 0.0,
            'motion_angle_std': 0.0,
            'frame_diff_mean': 0.0,
            'frame_diff_std': 0.0,
            'motion_density': 0.0
        }
    
    def _get_default_shape_features(self):
        """Default shape features when no contours found."""
        return {
            'object_area': 0.0,
            'object_perimeter': 0.0,
            'aspect_ratio': 0.0,
            'solidity': 0.0,
            'circularity': 0.0,
            'bounding_width': 0.0,
            'bounding_height': 0.0,
            'contour_count': 0
        }
    
    def _analyze_features(self, features):
        """Analyze extracted features."""
        if not features:
            return {'error': 'No features extracted'}
        
        # Convert to numpy array
        feature_names = list(features[0].keys())
        feature_matrix = np.array([[f[name] for name in feature_names] for f in features])
        
        # Principal Component Analysis
        pca = PCA(n_components=min(10, len(feature_names)))
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        pca_features = pca.fit_transform(feature_matrix_scaled)
        
        # Feature importance analysis
        feature_variance = np.var(feature_matrix_scaled, axis=0)
        feature_importance = feature_variance / np.sum(feature_variance)
        
        # Find most discriminative features
        important_features = sorted(zip(feature_names, feature_importance), 
                                  key=lambda x: x[1], reverse=True)
        
        return {
            'feature_names': feature_names,
            'feature_matrix_shape': feature_matrix.shape,
            'pca_components': pca.components_.tolist(),
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'feature_importance': dict(important_features),
            'most_important_features': important_features[:5]
        }
    
    def _detect_anomalies(self, features):
        """Detect anomalous frames using machine learning."""
        if len(features) < 10:
            return {'error': 'Insufficient data for anomaly detection'}
        
        # Convert to numpy array
        feature_names = list(features[0].keys())
        feature_matrix = np.array([[f[name] for name in feature_names] for f in features])
        
        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Detect anomalies
        anomaly_labels = self.anomaly_detector.fit_predict(feature_matrix_scaled)
        anomaly_scores = self.anomaly_detector.score_samples(feature_matrix_scaled)
        
        # Find anomalous frames
        anomalous_frames = np.where(anomaly_labels == -1)[0]
        
        return {
            'anomalous_frames': anomalous_frames.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_count': len(anomalous_frames),
            'anomaly_percentage': float(len(anomalous_frames) / len(features) * 100)
        }
    
    def _cluster_patterns(self, features):
        """Cluster similar patterns in the data."""
        if len(features) < 5:
            return {'error': 'Insufficient data for clustering'}
        
        # Convert to numpy array
        feature_names = list(features[0].keys())
        feature_matrix = np.array([[f[name] for name in feature_names] for f in features])
        
        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # DBSCAN clustering for finding dense regions
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        dbscan_labels = dbscan.fit_predict(feature_matrix_scaled)
        
        # K-means clustering for general grouping
        n_clusters = min(5, len(features) // 3)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(feature_matrix_scaled)
        else:
            kmeans_labels = np.zeros(len(features))
        
        # Analyze clusters
        unique_dbscan = np.unique(dbscan_labels)
        unique_kmeans = np.unique(kmeans_labels)
        
        return {
            'dbscan_labels': dbscan_labels.tolist(),
            'kmeans_labels': kmeans_labels.tolist(),
            'dbscan_clusters': len(unique_dbscan[unique_dbscan != -1]),
            'kmeans_clusters': len(unique_kmeans),
            'noise_points': int(np.sum(dbscan_labels == -1)),
            'cluster_analysis': self._analyze_cluster_characteristics(feature_matrix_scaled, kmeans_labels, feature_names)
        }
    
    def _classify_objects(self, frames):
        """Classify objects in frames using simple heuristics."""
        classifications = []
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Object detection
            object_properties = self._analyze_object_properties(gray)
            
            # Simple classification based on properties
            classification = self._classify_based_on_properties(object_properties)
            
            classifications.append({
                'frame': i,
                'properties': object_properties,
                'classification': classification
            })
        
        return classifications
    
    def _analyze_behavior_patterns(self, features, metadata):
        """Analyze behavioral patterns over time."""
        if len(features) < 10:
            return {'error': 'Insufficient data for behavior analysis'}
        
        # Extract time series of key features
        motion_series = [f['motion_magnitude_mean'] for f in features]
        intensity_series = [f['mean_intensity'] for f in features]
        area_series = [f['object_area'] for f in features]
        
        # Analyze patterns
        behavior_analysis = {
            'motion_pattern': self._analyze_time_series_pattern(motion_series),
            'intensity_pattern': self._analyze_time_series_pattern(intensity_series),
            'size_pattern': self._analyze_time_series_pattern(area_series),
            'correlation_analysis': self._analyze_feature_correlations(features)
        }
        
        return behavior_analysis
    
    def _analyze_frame_similarity(self, features):
        """Analyze similarity between frames."""
        if len(features) < 2:
            return {'error': 'Insufficient data for similarity analysis'}
        
        # Convert to numpy array
        feature_names = list(features[0].keys())
        feature_matrix = np.array([[f[name] for name in feature_names] for f in features])
        
        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(feature_matrix_scaled, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Find most similar and dissimilar frame pairs
        np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-similarity
        
        min_distance_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        max_distance_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        
        return {
            'distance_matrix': distance_matrix.tolist(),
            'most_similar_frames': [int(min_distance_idx[0]), int(min_distance_idx[1])],
            'most_dissimilar_frames': [int(max_distance_idx[0]), int(max_distance_idx[1])],
            'average_similarity': float(np.mean(distances)),
            'similarity_std': float(np.std(distances))
        }
    
    # Helper methods
    def _calculate_skewness(self, image):
        """Calculate skewness of image intensity distribution."""
        flat = image.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        if std == 0:
            return 0
        return np.mean(((flat - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, image):
        """Calculate kurtosis of image intensity distribution."""
        flat = image.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        if std == 0:
            return 0
        return np.mean(((flat - mean) / std) ** 4) - 3
    
    def _calculate_lbp(self, image):
        """Calculate simplified Local Binary Pattern."""
        # Simplified LBP implementation
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        return lbp
    
    def _calculate_color_entropy(self, color_image):
        """Calculate color entropy."""
        # Convert to 1D and calculate histogram
        flat = color_image.reshape(-1, 3)
        
        # Simple entropy calculation
        unique_colors = np.unique(flat, axis=0)
        entropy = len(unique_colors) / len(flat)
        
        return entropy
    
    def _calculate_spectral_centroid(self, magnitude_spectrum):
        """Calculate spectral centroid."""
        # Flatten spectrum and calculate weighted average
        flat_spectrum = magnitude_spectrum.flatten()
        frequencies = np.arange(len(flat_spectrum))
        
        if np.sum(flat_spectrum) == 0:
            return 0
        
        centroid = np.sum(frequencies * flat_spectrum) / np.sum(flat_spectrum)
        return centroid
    
    def _analyze_cluster_characteristics(self, feature_matrix, labels, feature_names):
        """Analyze characteristics of each cluster."""
        unique_labels = np.unique(labels)
        cluster_chars = {}
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = feature_matrix[cluster_mask]
            
            if len(cluster_data) > 0:
                cluster_chars[f'cluster_{label}'] = {
                    'size': len(cluster_data),
                    'centroid': np.mean(cluster_data, axis=0).tolist(),
                    'variance': np.var(cluster_data, axis=0).tolist()
                }
        
        return cluster_chars
    
    def _analyze_object_properties(self, gray_image):
        """Analyze properties of objects in image."""
        # Basic object detection and property analysis
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'no_object_detected': True}
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'width': float(w),
            'height': float(h),
            'aspect_ratio': float(w/h) if h > 0 else 0,
            'center_x': float(x + w/2),
            'center_y': float(y + h/2)
        }
    
    def _classify_based_on_properties(self, properties):
        """Simple classification based on object properties."""
        if properties.get('no_object_detected'):
            return 'no_object'
        
        area = properties.get('area', 0)
        aspect_ratio = properties.get('aspect_ratio', 0)
        
        # Simple heuristic classification
        if area < 100:
            return 'small_object'
        elif area > 5000:
            return 'large_object'
        elif aspect_ratio > 3:
            return 'elongated_object'
        elif 0.8 <= aspect_ratio <= 1.2:
            return 'circular_object'
        else:
            return 'irregular_object'
    
    def _analyze_time_series_pattern(self, series):
        """Analyze patterns in time series data."""
        if len(series) < 5:
            return {'insufficient_data': True}
        
        series_array = np.array(series)
        
        # Basic statistics
        trend = np.polyfit(range(len(series)), series, 1)[0]  # Linear trend
        volatility = np.std(series_array)
        
        # Detect cycles
        autocorr = np.correlate(series_array, series_array, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks in autocorrelation (potential cycles)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr[1:], height=0.5 * np.max(autocorr))
        
        return {
            'trend': float(trend),
            'volatility': float(volatility),
            'mean': float(np.mean(series_array)),
            'range': float(np.max(series_array) - np.min(series_array)),
            'potential_cycles': (peaks + 1).tolist() if len(peaks) > 0 else []
        }
    
    def _analyze_feature_correlations(self, features):
        """Analyze correlations between features."""
        feature_names = list(features[0].keys())
        feature_matrix = np.array([[f[name] for name in feature_names] for f in features])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(feature_matrix.T)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr_val = corr_matrix[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': float(corr_val)
                    })
        
        return {
            'correlation_matrix': corr_matrix.tolist(),
            'high_correlations': high_corr_pairs,
            'feature_names': feature_names
        }