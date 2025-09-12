#!/usr/bin/env python3
"""
Trajectory Prediction Component
==============================

Predicts future movement paths and analyzes trajectory patterns using
physics models, machine learning, and pattern recognition techniques.
"""

import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)

class TrajectoryPredictor:
    """Predicts and analyzes object trajectory patterns."""
    
    def __init__(self, config):
        """Initialize trajectory predictor."""
        self.config = config
        self.prediction_horizon = config.get('trajectory', {}).get('prediction_horizon', 10)  # frames
        self.physics_models = ['linear', 'parabolic', 'orbital', 'ballistic']
        
    def analyze(self, frames, metadata, analysis_results=None):
        """Analyze and predict trajectory patterns."""
        logger.info("Starting trajectory prediction analysis...")
        
        # Extract motion data
        motion_data = self._extract_motion_data(frames, analysis_results)
        
        if not motion_data or len(motion_data) < 3:
            return {'error': 'Insufficient motion data for trajectory analysis'}
        
        results = {
            'trajectory_extraction': motion_data,
            'physics_model_fitting': self._fit_physics_models(motion_data, metadata),
            'ml_trajectory_prediction': self._ml_trajectory_prediction(motion_data),
            'pattern_analysis': self._analyze_trajectory_patterns(motion_data),
            'predictability_analysis': self._analyze_predictability(motion_data),
            'anomaly_detection': self._detect_trajectory_anomalies(motion_data),
            'future_position_prediction': self._predict_future_positions(motion_data, metadata),
            'behavior_classification': self._classify_trajectory_behavior(motion_data),
            'turning_point_analysis': self._analyze_turning_points(motion_data),
            'velocity_profile_analysis': self._analyze_velocity_profile(motion_data, metadata)
        }
        
        # Calculate trajectory confidence
        results['trajectory_confidence'] = self._calculate_trajectory_confidence(results)
        
        return results
    
    def _extract_motion_data(self, frames, analysis_results):
        """Extract motion trajectory data from frames or analysis results."""
        if analysis_results and 'motion' in analysis_results:
            # Use existing motion analysis if available
            return self._extract_from_analysis_results(analysis_results['motion'])
        else:
            # Extract motion data directly from frames
            return self._extract_from_frames(frames)
    
    def _extract_from_frames(self, frames):
        """Extract motion data directly from video frames."""
        trajectory_points = []
        
        # Simple object tracking
        tracker = cv2.TrackerCSRT_create()
        
        # Initialize tracker on first frame
        first_frame = frames[0]
        gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect initial object
        bbox = self._detect_initial_object(first_frame)
        if bbox is None:
            return []
        
        ok = tracker.init(first_frame, bbox)
        if not ok:
            return []
        
        # Track through frames
        for i, frame in enumerate(frames):
            ok, bbox = tracker.update(frame)
            
            if ok:
                x, y, w, h = bbox
                center_x = x + w/2
                center_y = y + h/2
                
                trajectory_points.append({
                    'frame': i,
                    'x': center_x,
                    'y': center_y,
                    'bbox': bbox,
                    'timestamp': i / 30.0  # Assume 30 fps
                })
            else:
                # Lost tracking
                break
        
        return trajectory_points
    
    def _fit_physics_models(self, motion_data, metadata):
        """Fit various physics models to the trajectory."""
        if len(motion_data) < 4:
            return {'error': 'Insufficient data for physics model fitting'}
        
        # Extract coordinates and times
        times = np.array([point['timestamp'] for point in motion_data])
        x_coords = np.array([point['x'] for point in motion_data])
        y_coords = np.array([point['y'] for point in motion_data])
        
        physics_fits = {}
        
        # Linear motion model
        physics_fits['linear'] = self._fit_linear_model(times, x_coords, y_coords)
        
        # Parabolic motion model (projectile)
        physics_fits['parabolic'] = self._fit_parabolic_model(times, x_coords, y_coords)
        
        # Circular/orbital motion model
        physics_fits['circular'] = self._fit_circular_model(times, x_coords, y_coords)
        
        # Ballistic motion model
        physics_fits['ballistic'] = self._fit_ballistic_model(times, x_coords, y_coords, metadata)
        
        # Harmonic motion model
        physics_fits['harmonic'] = self._fit_harmonic_model(times, x_coords, y_coords)
        
        # Evaluate model fits
        best_model = self._evaluate_model_fits(physics_fits, times, x_coords, y_coords)
        
        return {
            'model_fits': physics_fits,
            'best_model': best_model,
            'physics_compliance': self._assess_physics_compliance(physics_fits, best_model)
        }
    
    def _ml_trajectory_prediction(self, motion_data):
        """Use machine learning for trajectory prediction."""
        if len(motion_data) < 5:
            return {'error': 'Insufficient data for ML prediction'}
        
        # Prepare features
        features, targets = self._prepare_ml_features(motion_data)
        
        if len(features) < 3:
            return {'error': 'Insufficient feature data for ML'}
        
        # Polynomial regression
        poly_prediction = self._polynomial_regression_prediction(features, targets)
        
        # Linear regression with velocity features
        velocity_prediction = self._velocity_based_prediction(motion_data)
        
        # Spline interpolation prediction
        spline_prediction = self._spline_prediction(motion_data)
        
        # Ensemble prediction
        ensemble_prediction = self._ensemble_prediction(
            poly_prediction, velocity_prediction, spline_prediction
        )
        
        return {
            'polynomial_prediction': poly_prediction,
            'velocity_prediction': velocity_prediction,
            'spline_prediction': spline_prediction,
            'ensemble_prediction': ensemble_prediction,
            'prediction_confidence': self._calculate_ml_confidence(
                poly_prediction, velocity_prediction, spline_prediction
            )
        }
    
    def _analyze_trajectory_patterns(self, motion_data):
        """Analyze patterns in the trajectory."""
        if len(motion_data) < 3:
            return {'error': 'Insufficient data for pattern analysis'}
        
        # Extract coordinates
        x_coords = np.array([point['x'] for point in motion_data])
        y_coords = np.array([point['y'] for point in motion_data])
        times = np.array([point['timestamp'] for point in motion_data])
        
        pattern_analysis = {
            'linearity': self._measure_trajectory_linearity(x_coords, y_coords),
            'curvature_analysis': self._analyze_curvature(x_coords, y_coords),
            'periodicity': self._detect_periodic_patterns(x_coords, y_coords, times),
            'symmetry': self._analyze_trajectory_symmetry(x_coords, y_coords),
            'complexity': self._measure_trajectory_complexity(x_coords, y_coords),
            'smoothness': self._measure_trajectory_smoothness(x_coords, y_coords),
            'turning_patterns': self._analyze_turning_patterns(x_coords, y_coords),
            'spiral_detection': self._detect_spiral_patterns(x_coords, y_coords)
        }
        
        return pattern_analysis
    
    def _analyze_predictability(self, motion_data):
        """Analyze how predictable the trajectory is."""
        if len(motion_data) < 6:
            return {'error': 'Insufficient data for predictability analysis'}
        
        # Use partial data to predict and compare with actual
        split_point = len(motion_data) // 2
        training_data = motion_data[:split_point]
        test_data = motion_data[split_point:]
        
        # Make predictions based on training data
        predictions = self._make_predictions_from_partial_data(training_data, len(test_data))
        
        # Compare predictions with actual trajectory
        prediction_errors = self._calculate_prediction_errors(predictions, test_data)
        
        # Analyze error patterns
        error_analysis = self._analyze_prediction_errors(prediction_errors)
        
        return {
            'prediction_errors': prediction_errors,
            'error_analysis': error_analysis,
            'predictability_score': self._calculate_predictability_score(prediction_errors),
            'chaos_indicators': self._detect_chaotic_behavior(motion_data),
            'determinism_score': self._calculate_determinism_score(prediction_errors)
        }
    
    def _detect_trajectory_anomalies(self, motion_data):
        """Detect anomalies in the trajectory."""
        anomalies = []
        
        if len(motion_data) < 4:
            return {'anomalies': [], 'anomaly_count': 0}
        
        # Extract coordinates and times
        x_coords = np.array([point['x'] for point in motion_data])
        y_coords = np.array([point['y'] for point in motion_data])
        times = np.array([point['timestamp'] for point in motion_data])
        
        # Detect sudden direction changes
        direction_anomalies = self._detect_sudden_direction_changes(x_coords, y_coords, times)
        anomalies.extend(direction_anomalies)
        
        # Detect speed anomalies
        speed_anomalies = self._detect_speed_anomalies(x_coords, y_coords, times)
        anomalies.extend(speed_anomalies)
        
        # Detect impossible accelerations
        acceleration_anomalies = self._detect_acceleration_anomalies(x_coords, y_coords, times)
        anomalies.extend(acceleration_anomalies)
        
        # Detect teleportation (discontinuous jumps)
        teleportation_anomalies = self._detect_teleportation(x_coords, y_coords, times)
        anomalies.extend(teleportation_anomalies)
        
        # Detect anti-gravitational behavior
        antigrav_anomalies = self._detect_antigravitational_behavior(x_coords, y_coords, times)
        anomalies.extend(antigrav_anomalies)
        
        return {
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'anomaly_severity': self._calculate_anomaly_severity(anomalies),
            'physics_violations': self._categorize_physics_violations(anomalies)
        }
    
    def _predict_future_positions(self, motion_data, metadata):
        """Predict future positions based on current trajectory."""
        if len(motion_data) < 3:
            return {'error': 'Insufficient data for future prediction'}
        
        # Use best fitting model to predict
        physics_fits = self._fit_physics_models(motion_data, metadata)
        best_model = physics_fits.get('best_model', {})
        
        if not best_model or 'model_type' not in best_model:
            return {'error': 'No suitable model for prediction'}
        
        # Generate future predictions
        current_time = motion_data[-1]['timestamp']
        prediction_times = np.linspace(
            current_time + 0.1, 
            current_time + self.prediction_horizon * 0.1, 
            self.prediction_horizon
        )
        
        future_positions = []
        
        for t in prediction_times:
            predicted_pos = self._predict_position_at_time(t, best_model)
            
            future_positions.append({
                'timestamp': float(t),
                'predicted_x': float(predicted_pos[0]),
                'predicted_y': float(predicted_pos[1]),
                'confidence': self._calculate_position_confidence(t, current_time, best_model)
            })
        
        return {
            'future_positions': future_positions,
            'prediction_model': best_model['model_type'],
            'prediction_horizon_seconds': float(self.prediction_horizon * 0.1),
            'prediction_reliability': self._assess_prediction_reliability(best_model, motion_data)
        }
    
    def _classify_trajectory_behavior(self, motion_data):
        """Classify the type of trajectory behavior."""
        if len(motion_data) < 3:
            return {'classification': 'insufficient_data'}
        
        # Extract motion characteristics
        x_coords = np.array([point['x'] for point in motion_data])
        y_coords = np.array([point['y'] for point in motion_data])
        times = np.array([point['timestamp'] for point in motion_data])
        
        # Calculate motion metrics
        linearity = self._measure_trajectory_linearity(x_coords, y_coords)
        curvature = self._calculate_average_curvature(x_coords, y_coords)
        speed_variation = self._calculate_speed_variation(x_coords, y_coords, times)
        
        # Classify behavior
        if linearity > 0.9:
            behavior_type = 'linear_motion'
        elif curvature > 0.1:
            if self._detect_spiral_patterns(x_coords, y_coords)['spiral_detected']:
                behavior_type = 'spiral_motion'
            else:
                behavior_type = 'curved_motion'
        elif speed_variation > 0.5:
            behavior_type = 'erratic_motion'
        elif self._detect_periodic_patterns(x_coords, y_coords, times)['periodic']:
            behavior_type = 'periodic_motion'
        elif self._detect_hovering_behavior(x_coords, y_coords, times):
            behavior_type = 'hovering'
        else:
            behavior_type = 'complex_motion'
        
        return {
            'classification': behavior_type,
            'confidence': self._calculate_classification_confidence(
                behavior_type, linearity, curvature, speed_variation
            ),
            'characteristics': {
                'linearity': float(linearity),
                'curvature': float(curvature),
                'speed_variation': float(speed_variation)
            }
        }
    
    def _analyze_turning_points(self, motion_data):
        """Analyze turning points and direction changes."""
        if len(motion_data) < 5:
            return {'turning_points': [], 'turning_point_count': 0}
        
        x_coords = np.array([point['x'] for point in motion_data])
        y_coords = np.array([point['y'] for point in motion_data])
        times = np.array([point['timestamp'] for point in motion_data])
        
        # Calculate direction vectors
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        dt = np.diff(times)
        
        # Calculate angles
        angles = np.arctan2(dy, dx)
        angle_changes = np.diff(angles)
        
        # Wrap angle changes to [-π, π]
        angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
        
        # Detect significant turning points
        turning_threshold = np.pi / 6  # 30 degrees
        turning_indices = np.where(np.abs(angle_changes) > turning_threshold)[0]
        
        turning_points = []
        for idx in turning_indices:
            turning_points.append({
                'frame': idx + 2,  # +2 because of double diff
                'timestamp': float(times[idx + 2]),
                'position': [float(x_coords[idx + 2]), float(y_coords[idx + 2])],
                'angle_change_degrees': float(np.degrees(angle_changes[idx])),
                'turn_sharpness': self._classify_turn_sharpness(angle_changes[idx])
            })
        
        return {
            'turning_points': turning_points,
            'turning_point_count': len(turning_points),
            'average_turn_angle': float(np.mean(np.abs(angle_changes[turning_indices]))) if len(turning_indices) > 0 else 0,
            'turn_frequency': float(len(turning_points) / (times[-1] - times[0])) if len(turning_points) > 0 else 0
        }
    
    def _analyze_velocity_profile(self, motion_data, metadata):
        """Analyze velocity profile over time."""
        if len(motion_data) < 3:
            return {'error': 'Insufficient data for velocity analysis'}
        
        x_coords = np.array([point['x'] for point in motion_data])
        y_coords = np.array([point['y'] for point in motion_data])
        times = np.array([point['timestamp'] for point in motion_data])
        
        # Calculate velocities
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        dt = np.diff(times)
        
        # Avoid division by zero
        dt = np.where(dt == 0, 1e-6, dt)
        
        vx = dx / dt
        vy = dy / dt
        speeds = np.sqrt(vx**2 + vy**2)
        
        # Calculate accelerations
        if len(speeds) > 1:
            accelerations = np.diff(speeds) / dt[1:]
        else:
            accelerations = np.array([])
        
        return {
            'velocity_profile': {
                'times': times[1:].tolist(),
                'speeds': speeds.tolist(),
                'velocities_x': vx.tolist(),
                'velocities_y': vy.tolist()
            },
            'acceleration_profile': {
                'times': times[2:].tolist() if len(accelerations) > 0 else [],
                'accelerations': accelerations.tolist()
            },
            'velocity_statistics': {
                'max_speed': float(np.max(speeds)),
                'min_speed': float(np.min(speeds)),
                'average_speed': float(np.mean(speeds)),
                'speed_std': float(np.std(speeds))
            },
            'acceleration_statistics': {
                'max_acceleration': float(np.max(accelerations)) if len(accelerations) > 0 else 0,
                'min_acceleration': float(np.min(accelerations)) if len(accelerations) > 0 else 0,
                'average_acceleration': float(np.mean(accelerations)) if len(accelerations) > 0 else 0
            }
        }
    
    # Helper methods for trajectory analysis
    def _detect_initial_object(self, frame):
        """Detect initial object for tracking."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple contour-based detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Return bounding box
        return (x, y, w, h)
    
    def _fit_linear_model(self, times, x_coords, y_coords):
        """Fit linear motion model."""
        try:
            # Linear regression for x and y
            x_poly = np.polyfit(times, x_coords, 1)
            y_poly = np.polyfit(times, y_coords, 1)
            
            # Calculate R-squared
            x_pred = np.polyval(x_poly, times)
            y_pred = np.polyval(y_poly, times)
            
            x_r2 = 1 - np.sum((x_coords - x_pred)**2) / np.sum((x_coords - np.mean(x_coords))**2)
            y_r2 = 1 - np.sum((y_coords - y_pred)**2) / np.sum((y_coords - np.mean(y_coords))**2)
            
            return {
                'x_coefficients': x_poly.tolist(),
                'y_coefficients': y_poly.tolist(),
                'x_r_squared': float(x_r2),
                'y_r_squared': float(y_r2),
                'avg_r_squared': float((x_r2 + y_r2) / 2),
                'model_type': 'linear'
            }
        except Exception as e:
            return {'error': f'Linear fit failed: {str(e)}'}
    
    def _fit_parabolic_model(self, times, x_coords, y_coords):
        """Fit parabolic motion model."""
        try:
            # Quadratic regression
            x_poly = np.polyfit(times, x_coords, 2)
            y_poly = np.polyfit(times, y_coords, 2)
            
            # Calculate R-squared
            x_pred = np.polyval(x_poly, times)
            y_pred = np.polyval(y_poly, times)
            
            x_r2 = 1 - np.sum((x_coords - x_pred)**2) / np.sum((x_coords - np.mean(x_coords))**2)
            y_r2 = 1 - np.sum((y_coords - y_pred)**2) / np.sum((y_coords - np.mean(y_coords))**2)
            
            return {
                'x_coefficients': x_poly.tolist(),
                'y_coefficients': y_poly.tolist(),
                'x_r_squared': float(x_r2),
                'y_r_squared': float(y_r2),
                'avg_r_squared': float((x_r2 + y_r2) / 2),
                'model_type': 'parabolic'
            }
        except Exception as e:
            return {'error': f'Parabolic fit failed: {str(e)}'}
    
    def _measure_trajectory_linearity(self, x_coords, y_coords):
        """Measure how linear the trajectory is."""
        if len(x_coords) < 3:
            return 0.0
        
        # Fit a line to the trajectory
        try:
            # Use least squares to fit line
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            slope, intercept = np.linalg.lstsq(A, y_coords, rcond=None)[0]
            
            # Calculate predicted y values
            y_pred = slope * x_coords + intercept
            
            # Calculate R-squared
            ss_res = np.sum((y_coords - y_pred) ** 2)
            ss_tot = np.sum((y_coords - np.mean(y_coords)) ** 2)
            
            if ss_tot == 0:
                return 1.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, r_squared)
            
        except Exception:
            return 0.0
    
    def _detect_sudden_direction_changes(self, x_coords, y_coords, times):
        """Detect sudden direction changes."""
        if len(x_coords) < 4:
            return []
        
        # Calculate velocity vectors
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        dt = np.diff(times)
        
        # Avoid division by zero
        dt = np.where(dt == 0, 1e-6, dt)
        
        vx = dx / dt
        vy = dy / dt
        
        # Calculate direction changes
        direction_changes = []
        for i in range(1, len(vx)):
            # Angle between consecutive velocity vectors
            v1 = np.array([vx[i-1], vy[i-1]])
            v2 = np.array([vx[i], vy[i]])
            
            # Calculate angle
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                # Detect sudden changes (> 90 degrees)
                if angle > np.pi / 2:
                    direction_changes.append({
                        'frame': i + 1,
                        'timestamp': float(times[i + 1]),
                        'angle_change_radians': float(angle),
                        'angle_change_degrees': float(np.degrees(angle)),
                        'type': 'sudden_direction_change'
                    })
        
        return direction_changes