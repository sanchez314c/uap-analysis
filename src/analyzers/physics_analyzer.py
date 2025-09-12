#!/usr/bin/env python3
"""
Physics Analysis Component
==========================

Analyzes physical properties and behavior patterns that may indicate
non-conventional propulsion or physics-defying characteristics.
"""

import cv2
import numpy as np
from scipy import optimize
import logging

logger = logging.getLogger(__name__)

class PhysicsAnalyzer:
    """Analyzes physical behavior patterns and anomalies."""
    
    def __init__(self, config):
        """Initialize physics analyzer."""
        self.config = config
        self.gravity = 9.81  # m/s²
        
    def analyze(self, frames, metadata):
        """Analyze physics patterns in video frames."""
        logger.info("Starting physics analysis...")
        
        # Extract motion data first
        motion_data = self._extract_motion_trajectory(frames)
        
        results = {
            'trajectory_analysis': self._analyze_trajectory(motion_data, metadata),
            'acceleration_analysis': self._analyze_acceleration(motion_data, metadata),
            'energy_analysis': self._analyze_energy_patterns(motion_data, metadata),
            'inertia_analysis': self._analyze_inertial_properties(motion_data, metadata),
            'gravitational_analysis': self._analyze_gravitational_effects(motion_data, metadata),
            'aerodynamic_analysis': self._analyze_aerodynamic_properties(motion_data, frames, metadata)
        }
        
        # Calculate physics anomaly score
        results['anomaly_score'] = self._calculate_physics_anomaly_score(results)
        
        return results
    
    def _extract_motion_trajectory(self, frames):
        """Extract object trajectory from frames."""
        trajectory = []
        centroids = []
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple object detection (can be enhanced)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (assumed to be the object)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
                    
                    trajectory.append({
                        'frame': i,
                        'position': (cx, cy),
                        'area': cv2.contourArea(largest_contour),
                        'contour': largest_contour
                    })
        
        return trajectory
    
    def _analyze_trajectory(self, motion_data, metadata):
        """Analyze trajectory patterns for physics anomalies."""
        if len(motion_data) < 3:
            return {'insufficient_data': True}
        
        positions = np.array([point['position'] for point in motion_data])
        
        # Fit polynomial to trajectory
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        t = np.arange(len(positions))
        
        # Fit different polynomial degrees
        fits = {}
        for degree in [1, 2, 3, 4]:
            try:
                poly_x = np.polyfit(t, x_coords, degree)
                poly_y = np.polyfit(t, y_coords, degree)
                
                # Calculate R-squared
                x_pred = np.polyval(poly_x, t)
                y_pred = np.polyval(poly_y, t)
                
                x_r2 = 1 - np.sum((x_coords - x_pred)**2) / np.sum((x_coords - np.mean(x_coords))**2)
                y_r2 = 1 - np.sum((y_coords - y_pred)**2) / np.sum((y_coords - np.mean(y_coords))**2)
                
                fits[f'degree_{degree}'] = {
                    'poly_x': poly_x,
                    'poly_y': poly_y,
                    'r2_x': x_r2,
                    'r2_y': y_r2,
                    'avg_r2': (x_r2 + y_r2) / 2
                }
            except:
                fits[f'degree_{degree}'] = {'error': True}
        
        # Analyze trajectory characteristics
        path_length = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
        displacement = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
        
        efficiency = displacement / path_length if path_length > 0 else 0
        
        # Detect sudden direction changes
        direction_changes = self._detect_direction_changes(positions)
        
        return {
            'polynomial_fits': fits,
            'path_length': float(path_length),
            'displacement': float(displacement),
            'path_efficiency': float(efficiency),
            'direction_changes': direction_changes,
            'trajectory_smoothness': self._calculate_trajectory_smoothness(positions)
        }
    
    def _analyze_acceleration(self, motion_data, metadata):
        """Analyze acceleration patterns."""
        if len(motion_data) < 3:
            return {'insufficient_data': True}
        
        positions = np.array([point['position'] for point in motion_data])
        fps = metadata.get('fps', 30)
        dt = 1.0 / fps
        
        # Calculate velocities
        velocities = np.diff(positions, axis=0) / dt
        
        # Calculate accelerations
        accelerations = np.diff(velocities, axis=0) / dt
        
        # Analyze acceleration magnitudes
        accel_magnitudes = np.sqrt(np.sum(accelerations**2, axis=1))
        
        # Detect instantaneous accelerations (physics anomalies)
        instantaneous_threshold = np.percentile(accel_magnitudes, 95)
        instantaneous_events = accel_magnitudes > instantaneous_threshold
        
        # Analyze g-forces
        pixel_to_meter = self._estimate_pixel_to_meter_ratio(motion_data, metadata)
        accel_ms2 = accel_magnitudes * pixel_to_meter
        g_forces = accel_ms2 / self.gravity
        
        return {
            'velocities': velocities.tolist(),
            'accelerations': accelerations.tolist(),
            'acceleration_magnitudes': accel_magnitudes.tolist(),
            'max_acceleration': float(np.max(accel_magnitudes)),
            'avg_acceleration': float(np.mean(accel_magnitudes)),
            'instantaneous_events': np.where(instantaneous_events)[0].tolist(),
            'max_g_force': float(np.max(g_forces)) if len(g_forces) > 0 else 0,
            'sudden_acceleration_count': int(np.sum(instantaneous_events))
        }
    
    def _analyze_energy_patterns(self, motion_data, metadata):
        """Analyze energy conservation and patterns."""
        if len(motion_data) < 3:
            return {'insufficient_data': True}
        
        positions = np.array([point['position'] for point in motion_data])
        areas = np.array([point['area'] for point in motion_data])
        
        fps = metadata.get('fps', 30)
        dt = 1.0 / fps
        
        # Estimate mass from area (assuming constant density)
        relative_mass = areas / np.mean(areas)
        
        # Calculate kinetic energy (relative)
        velocities = np.diff(positions, axis=0) / dt
        vel_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        
        kinetic_energy = 0.5 * relative_mass[1:] * vel_magnitudes**2
        
        # Calculate potential energy (relative to lowest point)
        heights = -positions[:, 1]  # Negative y is up
        min_height = np.min(heights)
        relative_heights = heights - min_height
        potential_energy = relative_mass * relative_heights
        
        # Total energy
        total_energy = kinetic_energy + potential_energy[1:]
        
        # Analyze energy conservation
        energy_variation = np.std(total_energy) / np.mean(total_energy) if np.mean(total_energy) > 0 else 0
        
        # Detect energy anomalies
        energy_jumps = np.abs(np.diff(total_energy))
        significant_jumps = energy_jumps > np.percentile(energy_jumps, 90)
        
        return {
            'kinetic_energy': kinetic_energy.tolist(),
            'potential_energy': potential_energy.tolist(),
            'total_energy': total_energy.tolist(),
            'energy_conservation_score': float(1 - energy_variation),
            'energy_anomalies': np.where(significant_jumps)[0].tolist(),
            'energy_anomaly_count': int(np.sum(significant_jumps))
        }
    
    def _analyze_inertial_properties(self, motion_data, metadata):
        """Analyze inertial behavior and momentum conservation."""
        if len(motion_data) < 4:
            return {'insufficient_data': True}
        
        positions = np.array([point['position'] for point in motion_data])
        areas = np.array([point['area'] for point in motion_data])
        
        fps = metadata.get('fps', 30)
        dt = 1.0 / fps
        
        # Calculate momentum
        velocities = np.diff(positions, axis=0) / dt
        relative_mass = areas[1:] / np.mean(areas)
        momentum = relative_mass[:, np.newaxis] * velocities
        
        # Analyze momentum changes
        momentum_changes = np.diff(momentum, axis=0)
        momentum_change_magnitudes = np.sqrt(np.sum(momentum_changes**2, axis=1))
        
        # Detect sudden momentum changes (possible external forces)
        threshold = np.percentile(momentum_change_magnitudes, 85)
        sudden_changes = momentum_change_magnitudes > threshold
        
        # Analyze rotational inertia (if object shape changes)
        area_changes = np.diff(areas)
        shape_stability = 1 - (np.std(areas) / np.mean(areas))
        
        return {
            'momentum': momentum.tolist(),
            'momentum_changes': momentum_changes.tolist(),
            'sudden_momentum_changes': np.where(sudden_changes)[0].tolist(),
            'shape_stability': float(shape_stability),
            'inertial_consistency': float(1 - np.std(momentum_change_magnitudes) / np.mean(momentum_change_magnitudes)) if np.mean(momentum_change_magnitudes) > 0 else 1
        }
    
    def _analyze_gravitational_effects(self, motion_data, metadata):
        """Analyze gravitational behavior."""
        if len(motion_data) < 5:
            return {'insufficient_data': True}
        
        positions = np.array([point['position'] for point in motion_data])
        
        fps = metadata.get('fps', 30)
        dt = 1.0 / fps
        
        # Analyze vertical motion (y-axis, assuming up is negative)
        y_positions = -positions[:, 1]  # Flip so up is positive
        
        # Calculate vertical acceleration
        y_velocities = np.diff(y_positions) / dt
        y_accelerations = np.diff(y_velocities) / dt
        
        # Expected gravitational acceleration (in pixels)
        pixel_to_meter = self._estimate_pixel_to_meter_ratio(motion_data, metadata)
        expected_gravity_pixels = -self.gravity / pixel_to_meter  # Negative because down is positive in image
        
        # Compare with observed vertical acceleration
        gravity_deviations = y_accelerations - expected_gravity_pixels
        
        # Detect anti-gravity behavior
        anti_gravity_events = y_accelerations > 0  # Positive acceleration upward
        
        # Analyze free fall behavior
        free_fall_score = self._analyze_free_fall_behavior(y_positions, dt)
        
        return {
            'vertical_accelerations': y_accelerations.tolist(),
            'gravity_deviations': gravity_deviations.tolist(),
            'anti_gravity_events': np.where(anti_gravity_events)[0].tolist(),
            'anti_gravity_count': int(np.sum(anti_gravity_events)),
            'free_fall_score': free_fall_score,
            'gravity_anomaly_score': float(np.std(gravity_deviations) / abs(expected_gravity_pixels)) if expected_gravity_pixels != 0 else 0
        }
    
    def _analyze_aerodynamic_properties(self, motion_data, frames, metadata):
        """Analyze aerodynamic behavior."""
        if len(motion_data) < 3:
            return {'insufficient_data': True}
        
        positions = np.array([point['position'] for point in motion_data])
        
        fps = metadata.get('fps', 30)
        dt = 1.0 / fps
        
        # Calculate velocities
        velocities = np.diff(positions, axis=0) / dt
        vel_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        
        # Analyze drag effects
        drag_analysis = self._analyze_drag_effects(vel_magnitudes, motion_data)
        
        # Detect sonic boom indicators
        sonic_indicators = self._detect_sonic_indicators(frames, vel_magnitudes, metadata)
        
        # Analyze lift characteristics
        lift_analysis = self._analyze_lift_characteristics(positions, velocities)
        
        return {
            'drag_analysis': drag_analysis,
            'sonic_indicators': sonic_indicators,
            'lift_analysis': lift_analysis,
            'velocity_profile': vel_magnitudes.tolist()
        }
    
    def _calculate_physics_anomaly_score(self, results):
        """Calculate overall physics anomaly score."""
        anomaly_factors = []
        
        # Trajectory anomalies
        if 'trajectory_analysis' in results and 'path_efficiency' in results['trajectory_analysis']:
            # Very high efficiency might indicate non-ballistic movement
            efficiency = results['trajectory_analysis']['path_efficiency']
            if efficiency > 0.95:
                anomaly_factors.append(0.3)
        
        # Acceleration anomalies
        if 'acceleration_analysis' in results and 'sudden_acceleration_count' in results['acceleration_analysis']:
            sudden_count = results['acceleration_analysis']['sudden_acceleration_count']
            if sudden_count > 0:
                anomaly_factors.append(min(sudden_count * 0.1, 0.4))
        
        # Energy anomalies
        if 'energy_analysis' in results and 'energy_conservation_score' in results['energy_analysis']:
            conservation = results['energy_analysis']['energy_conservation_score']
            if conservation < 0.8:
                anomaly_factors.append((1 - conservation) * 0.3)
        
        # Gravitational anomalies
        if 'gravitational_analysis' in results and 'anti_gravity_count' in results['gravitational_analysis']:
            anti_grav = results['gravitational_analysis']['anti_gravity_count']
            if anti_grav > 0:
                anomaly_factors.append(min(anti_grav * 0.2, 0.5))
        
        # Calculate weighted score
        if anomaly_factors:
            return min(sum(anomaly_factors), 1.0)
        else:
            return 0.0
    
    def _estimate_pixel_to_meter_ratio(self, motion_data, metadata):
        """Estimate pixel to meter conversion ratio."""
        # This is a simplified estimation - in practice, you'd need reference objects
        # or known distances in the frame
        frame_height = metadata.get('height', 720)
        estimated_field_of_view_meters = 100  # Assume 100m field of view
        return estimated_field_of_view_meters / frame_height
    
    def _detect_direction_changes(self, positions):
        """Detect sudden direction changes."""
        if len(positions) < 3:
            return []
        
        direction_changes = []
        for i in range(1, len(positions) - 1):
            v1 = positions[i] - positions[i-1]
            v2 = positions[i+1] - positions[i]
            
            # Calculate angle between vectors
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms > 0:
                cos_angle = dot_product / norms
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                
                # Detect sharp turns (> 45 degrees)
                if angle > np.pi / 4:
                    direction_changes.append({
                        'frame': i,
                        'angle_radians': float(angle),
                        'angle_degrees': float(np.degrees(angle))
                    })
        
        return direction_changes
    
    def _calculate_trajectory_smoothness(self, positions):
        """Calculate trajectory smoothness score."""
        if len(positions) < 3:
            return 1.0
        
        # Calculate second derivatives (curvature)
        second_derivatives = np.diff(positions, n=2, axis=0)
        curvatures = np.sqrt(np.sum(second_derivatives**2, axis=1))
        
        # Higher curvature variation indicates less smoothness
        if len(curvatures) > 0:
            smoothness = 1 / (1 + np.std(curvatures))
        else:
            smoothness = 1.0
        
        return float(smoothness)
    
    def _analyze_free_fall_behavior(self, y_positions, dt):
        """Analyze if motion follows free fall physics."""
        if len(y_positions) < 4:
            return 0.0
        
        # Fit parabolic equation: y = y0 + v0*t + 0.5*g*t^2
        t = np.arange(len(y_positions)) * dt
        
        try:
            # Polynomial fit (degree 2 for parabolic)
            coeffs = np.polyfit(t, y_positions, 2)
            y_pred = np.polyval(coeffs, t)
            
            # Calculate R-squared
            ss_res = np.sum((y_positions - y_pred) ** 2)
            ss_tot = np.sum((y_positions - np.mean(y_positions)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return float(max(0, r_squared))
        except:
            return 0.0
    
    def _analyze_drag_effects(self, velocities, motion_data):
        """Analyze drag and resistance effects."""
        # Simple drag analysis - look for velocity decrease patterns
        if len(velocities) < 3:
            return {'insufficient_data': True}
        
        # Calculate velocity changes
        vel_changes = np.diff(velocities)
        deceleration_events = vel_changes < 0
        
        # Analyze if deceleration follows expected drag patterns
        decel_consistency = np.std(vel_changes[deceleration_events]) if np.any(deceleration_events) else 0
        
        return {
            'deceleration_events': int(np.sum(deceleration_events)),
            'deceleration_consistency': float(decel_consistency),
            'drag_anomaly_score': float(1 - decel_consistency / np.mean(np.abs(vel_changes))) if np.mean(np.abs(vel_changes)) > 0 else 0
        }
    
    def _detect_sonic_indicators(self, frames, velocities, metadata):
        """Detect potential sonic boom or shock wave indicators."""
        # Look for visual disturbances that might indicate sonic effects
        sonic_events = []
        
        pixel_to_meter = self._estimate_pixel_to_meter_ratio([], metadata)
        fps = metadata.get('fps', 30)
        
        for i, vel in enumerate(velocities):
            # Convert velocity to m/s
            vel_ms = vel * pixel_to_meter * fps
            
            # Check if approaching or exceeding sound speed (343 m/s)
            if vel_ms > 300:  # Close to sonic
                # Look for visual disturbances in the frame
                if i + 1 < len(frames):
                    frame = frames[i + 1]
                    # Analyze for shock wave patterns (simplified)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    
                    # Look for radial patterns
                    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
                    
                    sonic_events.append({
                        'frame': i + 1,
                        'velocity_ms': float(vel_ms),
                        'potential_sonic': vel_ms > 343,
                        'shock_patterns': lines is not None and len(lines) > 10
                    })
        
        return sonic_events
    
    def _analyze_lift_characteristics(self, positions, velocities):
        """Analyze lift and aerodynamic behavior."""
        if len(positions) < 3:
            return {'insufficient_data': True}
        
        # Analyze vertical vs horizontal motion correlation
        vertical_motion = -np.diff(positions[:, 1])  # Up is positive
        horizontal_motion = np.diff(positions[:, 0])
        
        # Look for lift patterns (upward motion during forward motion)
        lift_events = []
        for i in range(len(vertical_motion)):
            if vertical_motion[i] > 0 and abs(horizontal_motion[i]) > 0:
                lift_ratio = vertical_motion[i] / abs(horizontal_motion[i])
                if lift_ratio > 0.1:  # Significant upward component
                    lift_events.append({
                        'frame': i,
                        'lift_ratio': float(lift_ratio)
                    })
        
        return {
            'lift_events': lift_events,
            'lift_event_count': len(lift_events),
            'avg_lift_ratio': float(np.mean([e['lift_ratio'] for e in lift_events])) if lift_events else 0
        }