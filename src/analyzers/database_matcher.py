#!/usr/bin/env python3
"""
Database Pattern Matching Component
===================================

Matches detected patterns against known phenomena databases including
aircraft signatures, natural phenomena, and previously documented UAP cases.
"""

import cv2
import numpy as np
import json
import sqlite3
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class DatabaseMatcher:
    """Pattern matching against known phenomena databases."""
    
    def __init__(self, config):
        """Initialize database matcher."""
        self.config = config
        self.db_path = Path(config.get('database', {}).get('path', 'data/phenomena_database.db'))
        self.similarity_threshold = config.get('database', {}).get('similarity_threshold', 0.7)
        
        # Initialize databases
        self._initialize_databases()
        
    def analyze(self, frames, metadata, analysis_results=None):
        """Match against known phenomena databases."""
        logger.info("Starting database pattern matching...")
        
        # Extract features for matching
        features = self._extract_matching_features(frames, analysis_results)
        
        results = {
            'aircraft_matching': self._match_aircraft_signatures(features, metadata),
            'natural_phenomena_matching': self._match_natural_phenomena(features, metadata),
            'uap_case_matching': self._match_uap_cases(features, metadata),
            'celestial_object_matching': self._match_celestial_objects(features, metadata),
            'atmospheric_phenomena_matching': self._match_atmospheric_phenomena(features, metadata),
            'technology_matching': self._match_known_technology(features, metadata),
            'similarity_scores': self._calculate_similarity_scores(features),
            'classification_confidence': self._calculate_classification_confidence(features)
        }
        
        # Generate overall classification
        results['overall_classification'] = self._generate_overall_classification(results)
        
        return results
    
    def _initialize_databases(self):
        """Initialize or create pattern databases."""
        # Create database if it doesn't exist
        self._create_database_schema()
        
        # Load reference patterns
        self._load_aircraft_signatures()
        self._load_natural_phenomena_patterns()
        self._load_uap_reference_cases()
        self._load_celestial_objects()
        self._load_atmospheric_patterns()
        self._load_technology_signatures()
    
    def _create_database_schema(self):
        """Create database schema for pattern storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Aircraft signatures table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS aircraft_signatures (
            id INTEGER PRIMARY KEY,
            aircraft_type TEXT,
            signature_features TEXT,
            motion_characteristics TEXT,
            visual_features TEXT,
            size_range TEXT,
            speed_range TEXT,
            altitude_range TEXT,
            confidence REAL
        )
        ''')
        
        # Natural phenomena table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS natural_phenomena (
            id INTEGER PRIMARY KEY,
            phenomenon_type TEXT,
            signature_features TEXT,
            environmental_conditions TEXT,
            duration_range TEXT,
            seasonal_correlation TEXT,
            geographic_correlation TEXT,
            confidence REAL
        )
        ''')
        
        # UAP reference cases table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS uap_cases (
            id INTEGER PRIMARY KEY,
            case_id TEXT,
            signature_features TEXT,
            motion_patterns TEXT,
            luminosity_patterns TEXT,
            environmental_conditions TEXT,
            classification TEXT,
            credibility_score REAL
        )
        ''')
        
        # Celestial objects table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS celestial_objects (
            id INTEGER PRIMARY KEY,
            object_type TEXT,
            signature_features TEXT,
            orbital_characteristics TEXT,
            brightness_patterns TEXT,
            visibility_conditions TEXT,
            motion_predictions TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _extract_matching_features(self, frames, analysis_results):
        """Extract features for database matching."""
        features = {
            'motion_features': self._extract_motion_features(analysis_results),
            'luminosity_features': self._extract_luminosity_features(analysis_results),
            'shape_features': self._extract_shape_features(frames),
            'color_features': self._extract_color_features(frames),
            'temporal_features': self._extract_temporal_features(analysis_results),
            'size_features': self._extract_size_features(frames),
            'behavior_features': self._extract_behavior_features(analysis_results)
        }
        
        return features
    
    def _match_aircraft_signatures(self, features, metadata):
        """Match against known aircraft signatures."""
        matches = []
        
        # Load aircraft database
        aircraft_db = self._load_aircraft_database()
        
        for aircraft in aircraft_db:
            similarity = self._calculate_aircraft_similarity(features, aircraft)
            
            if similarity > self.similarity_threshold:
                matches.append({
                    'aircraft_type': aircraft['type'],
                    'similarity_score': similarity,
                    'matching_features': self._identify_matching_features(features, aircraft),
                    'confidence': self._calculate_aircraft_confidence(similarity, features, aircraft)
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            'matches': matches[:5],  # Top 5 matches
            'best_match': matches[0] if matches else None,
            'aircraft_probability': self._calculate_aircraft_probability(matches)
        }
    
    def _match_natural_phenomena(self, features, metadata):
        """Match against natural phenomena patterns."""
        phenomena_matches = []
        
        # Check atmospheric phenomena
        atmospheric_matches = self._check_atmospheric_phenomena(features, metadata)
        phenomena_matches.extend(atmospheric_matches)
        
        # Check celestial phenomena
        celestial_matches = self._check_celestial_phenomena(features, metadata)
        phenomena_matches.extend(celestial_matches)
        
        # Check weather-related phenomena
        weather_matches = self._check_weather_phenomena(features, metadata)
        phenomena_matches.extend(weather_matches)
        
        # Check optical phenomena
        optical_matches = self._check_optical_phenomena(features, metadata)
        phenomena_matches.extend(optical_matches)
        
        return {
            'atmospheric_phenomena': atmospheric_matches,
            'celestial_phenomena': celestial_matches,
            'weather_phenomena': weather_matches,
            'optical_phenomena': optical_matches,
            'best_natural_explanation': self._get_best_natural_explanation(phenomena_matches)
        }
    
    def _match_uap_cases(self, features, metadata):
        """Match against documented UAP cases."""
        uap_matches = []
        
        # Load UAP case database
        uap_cases = self._load_uap_database()
        
        for case in uap_cases:
            similarity = self._calculate_uap_case_similarity(features, case)
            
            if similarity > 0.5:  # Lower threshold for UAP cases
                uap_matches.append({
                    'case_id': case['case_id'],
                    'similarity_score': similarity,
                    'case_classification': case['classification'],
                    'matching_patterns': self._identify_uap_matching_patterns(features, case),
                    'credibility_score': case['credibility_score']
                })
        
        # Sort by similarity and credibility
        uap_matches.sort(key=lambda x: x['similarity_score'] * x['credibility_score'], reverse=True)
        
        return {
            'similar_cases': uap_matches[:10],  # Top 10 similar cases
            'pattern_categories': self._categorize_uap_patterns(uap_matches),
            'historical_correlation': self._analyze_historical_correlation(uap_matches)
        }
    
    def _match_celestial_objects(self, features, metadata):
        """Match against celestial objects."""
        celestial_matches = []
        
        # Check satellites
        satellite_matches = self._check_satellite_signatures(features, metadata)
        
        # Check planets and stars
        stellar_matches = self._check_stellar_objects(features, metadata)
        
        # Check space debris
        debris_matches = self._check_space_debris(features, metadata)
        
        # Check ISS and other space stations
        station_matches = self._check_space_stations(features, metadata)
        
        return {
            'satellite_matches': satellite_matches,
            'stellar_matches': stellar_matches,
            'debris_matches': debris_matches,
            'space_station_matches': station_matches,
            'celestial_probability': self._calculate_celestial_probability(
                satellite_matches, stellar_matches, debris_matches, station_matches
            )
        }
    
    def _match_atmospheric_phenomena(self, features, metadata):
        """Match against atmospheric phenomena."""
        atmospheric_matches = []
        
        # Ball lightning
        ball_lightning = self._check_ball_lightning(features)
        if ball_lightning['probability'] > 0.3:
            atmospheric_matches.append(ball_lightning)
        
        # Plasma phenomena
        plasma_phenomena = self._check_plasma_phenomena(features)
        if plasma_phenomena['probability'] > 0.3:
            atmospheric_matches.append(plasma_phenomena)
        
        # St. Elmo's Fire
        st_elmos_fire = self._check_st_elmos_fire(features)
        if st_elmos_fire['probability'] > 0.3:
            atmospheric_matches.append(st_elmos_fire)
        
        # Atmospheric sprites/elves
        sprite_phenomena = self._check_sprite_phenomena(features)
        if sprite_phenomena['probability'] > 0.3:
            atmospheric_matches.append(sprite_phenomena)
        
        # Temperature inversions effects
        inversion_effects = self._check_inversion_effects(features)
        if inversion_effects['probability'] > 0.3:
            atmospheric_matches.append(inversion_effects)
        
        return {
            'atmospheric_matches': atmospheric_matches,
            'best_atmospheric_explanation': max(atmospheric_matches, 
                                              key=lambda x: x['probability']) if atmospheric_matches else None
        }
    
    def _match_known_technology(self, features, metadata):
        """Match against known technology signatures."""
        technology_matches = []
        
        # Drones/UAVs
        drone_matches = self._check_drone_signatures(features, metadata)
        technology_matches.extend(drone_matches)
        
        # Military aircraft
        military_matches = self._check_military_aircraft(features, metadata)
        technology_matches.extend(military_matches)
        
        # Experimental aircraft
        experimental_matches = self._check_experimental_aircraft(features, metadata)
        technology_matches.extend(experimental_matches)
        
        # Rockets and missiles
        rocket_matches = self._check_rocket_signatures(features, metadata)
        technology_matches.extend(rocket_matches)
        
        return {
            'drone_matches': [m for m in technology_matches if m['category'] == 'drone'],
            'military_matches': [m for m in technology_matches if m['category'] == 'military'],
            'experimental_matches': [m for m in technology_matches if m['category'] == 'experimental'],
            'rocket_matches': [m for m in technology_matches if m['category'] == 'rocket'],
            'technology_probability': self._calculate_technology_probability(technology_matches)
        }
    
    def _calculate_similarity_scores(self, features):
        """Calculate similarity scores across all categories."""
        similarity_scores = {}
        
        # Load reference feature vectors
        reference_vectors = self._load_reference_feature_vectors()
        
        # Convert features to vector
        feature_vector = self._features_to_vector(features)
        
        for category, ref_vectors in reference_vectors.items():
            similarities = []
            for ref_vector in ref_vectors:
                similarity = cosine_similarity([feature_vector], [ref_vector])[0][0]
                similarities.append(similarity)
            
            similarity_scores[category] = {
                'max_similarity': float(np.max(similarities)) if similarities else 0.0,
                'avg_similarity': float(np.mean(similarities)) if similarities else 0.0,
                'similarity_distribution': similarities
            }
        
        return similarity_scores
    
    def _calculate_classification_confidence(self, features):
        """Calculate confidence in classification."""
        confidence_factors = []
        
        # Feature completeness
        total_features = len([f for f in features.values() if f is not None])
        expected_features = 7  # motion, luminosity, shape, color, temporal, size, behavior
        completeness = total_features / expected_features
        confidence_factors.append(('completeness', completeness, 0.2))
        
        # Feature quality (simplified)
        quality_scores = []
        for feature_type, feature_data in features.items():
            if feature_data:
                # Simple quality metric based on data richness
                if isinstance(feature_data, dict):
                    quality = len(feature_data) / 10  # Normalize
                elif isinstance(feature_data, list):
                    quality = min(len(feature_data) / 20, 1.0)  # Normalize
                else:
                    quality = 0.5  # Default for simple values
                quality_scores.append(quality)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.5
        confidence_factors.append(('quality', avg_quality, 0.3))
        
        # Consistency across features
        consistency = self._calculate_feature_consistency(features)
        confidence_factors.append(('consistency', consistency, 0.3))
        
        # Uniqueness (how distinct the features are)
        uniqueness = self._calculate_feature_uniqueness(features)
        confidence_factors.append(('uniqueness', uniqueness, 0.2))
        
        # Calculate weighted confidence
        total_weight = sum(weight for _, _, weight in confidence_factors)
        weighted_confidence = sum(score * weight for _, score, weight in confidence_factors) / total_weight
        
        return {
            'overall_confidence': float(weighted_confidence),
            'confidence_factors': {name: score for name, score, _ in confidence_factors}
        }
    
    def _generate_overall_classification(self, results):
        """Generate overall classification based on all matching results."""
        classifications = []
        
        # Aircraft classification
        aircraft_matches = results.get('aircraft_matching', {}).get('matches', [])
        if aircraft_matches:
            best_aircraft = aircraft_matches[0]
            classifications.append({
                'category': 'aircraft',
                'specific_type': best_aircraft['aircraft_type'],
                'confidence': best_aircraft['confidence'],
                'probability': results['aircraft_matching']['aircraft_probability']
            })
        
        # Natural phenomena classification
        natural_explanation = results.get('natural_phenomena_matching', {}).get('best_natural_explanation')
        if natural_explanation:
            classifications.append({
                'category': 'natural_phenomenon',
                'specific_type': natural_explanation['type'],
                'confidence': natural_explanation['confidence'],
                'probability': natural_explanation['probability']
            })
        
        # Celestial object classification
        celestial_prob = results.get('celestial_object_matching', {}).get('celestial_probability', 0)
        if celestial_prob > 0.5:
            classifications.append({
                'category': 'celestial_object',
                'specific_type': 'satellite_or_celestial',
                'confidence': celestial_prob,
                'probability': celestial_prob
            })
        
        # Technology classification
        tech_prob = results.get('technology_matching', {}).get('technology_probability', 0)
        if tech_prob > 0.5:
            classifications.append({
                'category': 'known_technology',
                'specific_type': 'drone_or_aircraft',
                'confidence': tech_prob,
                'probability': tech_prob
            })
        
        # UAP classification (if no conventional explanation found)
        conventional_prob = max([c['probability'] for c in classifications]) if classifications else 0
        if conventional_prob < 0.7:
            uap_matches = results.get('uap_case_matching', {}).get('similar_cases', [])
            uap_prob = 1.0 - conventional_prob
            
            classifications.append({
                'category': 'unidentified_aerial_phenomenon',
                'specific_type': 'unknown',
                'confidence': uap_prob,
                'probability': uap_prob,
                'similar_cases': len(uap_matches)
            })
        
        # Sort by probability
        classifications.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'primary_classification': classifications[0] if classifications else None,
            'alternative_classifications': classifications[1:3] if len(classifications) > 1 else [],
            'all_classifications': classifications,
            'identification_confidence': classifications[0]['confidence'] if classifications else 0.0
        }
    
    # Helper methods for database operations and matching
    def _load_aircraft_database(self):
        """Load aircraft signature database."""
        # This would load from actual database in production
        return [
            {
                'type': 'commercial_airliner',
                'motion_pattern': 'linear_steady',
                'size_range': [50, 80],
                'speed_range': [200, 900],
                'lights': 'navigation_strobes'
            },
            {
                'type': 'general_aviation',
                'motion_pattern': 'variable_linear',
                'size_range': [5, 20],
                'speed_range': [50, 300],
                'lights': 'basic_navigation'
            },
            {
                'type': 'helicopter',
                'motion_pattern': 'hovering_capable',
                'size_range': [10, 30],
                'speed_range': [0, 250],
                'lights': 'rotating_beacon'
            },
            {
                'type': 'military_fighter',
                'motion_pattern': 'high_performance',
                'size_range': [15, 25],
                'speed_range': [200, 2000],
                'lights': 'minimal_or_none'
            }
        ]
    
    def _calculate_aircraft_similarity(self, features, aircraft):
        """Calculate similarity to aircraft signature."""
        similarity_score = 0.0
        factors = 0
        
        # Motion pattern similarity
        if 'motion_features' in features and features['motion_features']:
            motion_sim = self._compare_motion_patterns(features['motion_features'], aircraft['motion_pattern'])
            similarity_score += motion_sim
            factors += 1
        
        # Size similarity
        if 'size_features' in features and features['size_features']:
            size_sim = self._compare_size_ranges(features['size_features'], aircraft['size_range'])
            similarity_score += size_sim
            factors += 1
        
        # Light pattern similarity
        if 'luminosity_features' in features and features['luminosity_features']:
            light_sim = self._compare_light_patterns(features['luminosity_features'], aircraft['lights'])
            similarity_score += light_sim
            factors += 1
        
        return similarity_score / factors if factors > 0 else 0.0
    
    def _check_ball_lightning(self, features):
        """Check for ball lightning characteristics."""
        probability = 0.0
        
        # Ball lightning characteristics
        # - Spherical or oval shape
        # - Bright luminosity
        # - Erratic movement
        # - Short duration
        # - Associated with thunderstorms
        
        if 'shape_features' in features:
            shape_data = features['shape_features']
            if isinstance(shape_data, dict):
                circularity = shape_data.get('circularity', 0)
                if circularity > 0.7:
                    probability += 0.3
        
        if 'luminosity_features' in features:
            lum_data = features['luminosity_features']
            if isinstance(lum_data, dict):
                brightness = lum_data.get('average_brightness', 0)
                if brightness > 0.7:
                    probability += 0.2
        
        if 'motion_features' in features:
            motion_data = features['motion_features']
            if isinstance(motion_data, dict):
                erratic_motion = motion_data.get('direction_changes', 0)
                if erratic_motion > 5:
                    probability += 0.3
        
        return {
            'type': 'ball_lightning',
            'probability': min(probability, 1.0),
            'confidence': probability * 0.8,
            'explanation': 'Rare atmospheric electrical phenomenon'
        }
    
    def _features_to_vector(self, features):
        """Convert features dictionary to numerical vector."""
        vector = []
        
        # Motion features
        motion = features.get('motion_features', {})
        if isinstance(motion, dict):
            vector.extend([
                motion.get('average_speed', 0),
                motion.get('max_acceleration', 0),
                motion.get('direction_changes', 0),
                motion.get('path_efficiency', 0.5)
            ])
        else:
            vector.extend([0, 0, 0, 0.5])
        
        # Luminosity features
        luminosity = features.get('luminosity_features', {})
        if isinstance(luminosity, dict):
            vector.extend([
                luminosity.get('average_brightness', 0.5),
                luminosity.get('brightness_variation', 0),
                luminosity.get('pulse_frequency', 0)
            ])
        else:
            vector.extend([0.5, 0, 0])
        
        # Shape features
        shape = features.get('shape_features', {})
        if isinstance(shape, dict):
            vector.extend([
                shape.get('circularity', 0.5),
                shape.get('aspect_ratio', 1.0),
                shape.get('size_consistency', 0.5)
            ])
        else:
            vector.extend([0.5, 1.0, 0.5])
        
        # Normalize vector
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def _load_reference_feature_vectors(self):
        """Load reference feature vectors for each category."""
        # This would load from database in production
        return {
            'aircraft': [
                [0.5, 0.2, 0.1, 0.9, 0.3, 0.1, 0.0, 0.2, 1.5, 0.8],  # Commercial aircraft
                [0.3, 0.4, 0.3, 0.7, 0.4, 0.2, 0.1, 0.4, 1.2, 0.6],  # General aviation
                [0.2, 0.1, 0.5, 0.5, 0.5, 0.3, 0.2, 0.6, 1.0, 0.7],  # Helicopter
            ],
            'natural_phenomena': [
                [0.1, 0.1, 0.8, 0.3, 0.8, 0.6, 0.0, 0.9, 1.0, 0.4],  # Ball lightning
                [0.8, 0.0, 0.1, 0.9, 0.2, 0.1, 0.0, 0.3, 2.0, 0.9],  # Satellite
                [0.0, 0.0, 0.2, 0.8, 0.6, 0.4, 0.3, 0.7, 1.1, 0.5],  # Atmospheric phenomenon
            ],
            'celestial': [
                [0.9, 0.0, 0.0, 0.95, 0.1, 0.0, 0.0, 0.2, 1.0, 0.95],  # Satellite
                [0.8, 0.0, 0.0, 0.9, 0.3, 0.1, 0.0, 0.1, 1.0, 0.9],   # Planet
                [0.7, 0.0, 0.1, 0.85, 0.2, 0.05, 0.0, 0.15, 1.0, 0.85] # Star
            ]
        }
    
    # Additional helper methods would continue here...