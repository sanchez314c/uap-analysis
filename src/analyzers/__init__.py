"""
UAP Analysis Components
======================

This package contains modular analysis components for UAP video analysis.
Each analyzer focuses on a specific aspect of the analysis pipeline.
"""

# Core analyzers
from .motion_analyzer import MotionAnalyzer
from .luminosity_analyzer import LuminosityAnalyzer

# Advanced analyzers
from .atmospheric_analyzer import AtmosphericAnalyzer
from .physics_analyzer import PhysicsAnalyzer
from .signature_analyzer import SignatureAnalyzer
from .ml_classifier import MLClassifier
from .stereo_vision_analyzer import StereoVisionAnalyzer
from .environmental_analyzer import EnvironmentalAnalyzer
from .database_matcher import DatabaseMatcher
from .acoustic_analyzer import AcousticAnalyzer
from .trajectory_predictor import TrajectoryPredictor
from .multispectral_analyzer import MultiSpectralAnalyzer

__all__ = [
    # Core analyzers
    'MotionAnalyzer',
    'LuminosityAnalyzer',
    
    # Advanced analyzers  
    'AtmosphericAnalyzer',
    'PhysicsAnalyzer',
    'SignatureAnalyzer',
    'MLClassifier',
    'StereoVisionAnalyzer',
    'EnvironmentalAnalyzer',
    'DatabaseMatcher',
    'AcousticAnalyzer',
    'TrajectoryPredictor',
    'MultiSpectralAnalyzer'
]