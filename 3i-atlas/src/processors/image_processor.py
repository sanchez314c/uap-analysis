#!/usr/bin/env python3
"""
3I Atlas Image Analysis Suite
============================
Advanced astronomical image analysis system for investigating interstellar objects.
Forked from UAP Analysis Suite with specialized astronomical and forensic capabilities.

Author: Claude Code Analysis System
Date: October 12, 2025
Version: 1.0.0

Features:
- High-resolution TIFF image processing
- Multi-spectral analysis and signature detection
- Astronomical object classification and anomaly detection
- Photometric and morphological analysis
- Forensic image validation and manipulation detection
- Advanced visualization and reporting
"""

import cv2
import numpy as np
from scipy import signal, ndimage, stats
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from pathlib import Path
import json
import yaml
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

class AtlasImageProcessor:
    """Core image processing and enhancement engine for astronomical images"""

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def load_tiff(self, file_path):
        """Load high-resolution TIFF image with proper error handling"""
        try:
            # Use cv2 for TIFF loading with 16-bit support
            image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

            if image is None:
                raise ValueError(f"Could not load image: {file_path}")

            self.logger.info(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")

            # Convert to float64 for precision calculations
            if image.dtype == np.uint16:
                image = image.astype(np.float64) / 65535.0
            elif image.dtype == np.uint8:
                image = image.astype(np.float64) / 255.0
            else:
                image = image.astype(np.float64)

            return image

        except Exception as e:
            self.logger.error(f"Error loading TIFF: {e}")
            raise

    def enhance_image(self, image):
        """Multi-stage image enhancement optimized for astronomical objects"""
        enhanced = image.copy()

        # 1. Noise reduction using adaptive bilateral filter
        enhanced = self._adaptive_bilateral_filter(enhanced)

        # 2. Contrast enhancement using CLAHE
        enhanced = self._clahe_enhancement(enhanced)

        # 3. Sharpness enhancement using unsharp masking
        enhanced = self._unsharp_mask(enhanced)

        # 4. Background subtraction for astronomical objects
        enhanced = self._background_subtraction(enhanced)

        return enhanced

    def _adaptive_bilateral_filter(self, image, d=9, sigma_color=0.1, sigma_space=0.1):
        """Adaptive bilateral filter for noise preservation"""
        # Convert to 8-bit for OpenCV processing
        img_8bit = (image * 255).astype(np.uint8)

        if len(img_8bit.shape) == 3:
            filtered = cv2.bilateralFilter(img_8bit, d, sigma_color*255, sigma_space*255)
        else:
            filtered = cv2.bilateralFilter(img_8bit, d, sigma_color*255, sigma_space*255)

        return filtered.astype(np.float64) / 255.0

    def _clahe_enhancement(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Contrast Limited Adaptive Histogram Equalization"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if len(image.shape) == 3:
            # Process each channel independently
            enhanced = np.zeros_like(image)
            for i in range(image.shape[2]):
                channel_8bit = (image[:, :, i] * 255).astype(np.uint8)
                enhanced[:, :, i] = clahe.apply(channel_8bit).astype(np.float64) / 255.0
        else:
            img_8bit = (image * 255).astype(np.uint8)
            enhanced = clahe.apply(img_8bit).astype(np.float64) / 255.0

        return enhanced

    def _unsharp_mask(self, image, sigma=1.0, strength=1.5):
        """Unsharp masking for edge enhancement"""
        blurred = ndimage.gaussian_filter(image, sigma=sigma)
        sharpened = image + strength * (image - blurred)
        return np.clip(sharpened, 0, 1)

    def _background_subtraction(self, image, kernel_size=51):
        """Background subtraction using morphological operations"""
        if len(image.shape) == 3:
            # Convert to grayscale for background estimation
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Use large kernel for background estimation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Subtract background
        if len(image.shape) == 3:
            background_expanded = np.stack([background] * 3, axis=2)
            result = image - background_expanded
        else:
            result = image - background

        return np.clip(result, 0, 1)

    def extract_features(self, image):
        """Extract key features for analysis"""
        features = {}

        # 1. Edge detection using multiple methods
        features['edges_canny'] = self._multi_scale_canny(image)
        features['edges_sobel'] = self._sobel_edges(image)

        # 2. Texture features using GLCM-like analysis
        features['texture'] = self._texture_analysis(image)

        # 3. Gradient information
        features['gradients'] = self._gradient_analysis(image)

        # 4. Frequency domain features
        features['frequency'] = self._frequency_analysis(image)

        return features

    def _multi_scale_canny(self, image, scales=[0.5, 1.0, 2.0]):
        """Multi-scale Canny edge detection"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        edges_multi = []
        for scale in scales:
            # Resize image
            h, w = gray.shape
            resized = cv2.resize(gray, (int(w*scale), int(h*scale)))

            # Canny detection
            edges = cv2.Canny((resized * 255).astype(np.uint8), 50, 150)

            # Resize back
            edges_resized = cv2.resize(edges, (w, h))
            edges_multi.append(edges_resized.astype(np.float64) / 255.0)

        return np.stack(edges_multi, axis=2)

    def _sobel_edges(self, image):
        """Sobel edge detection with magnitude and direction"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x)

        return np.stack([magnitude, direction], axis=2)

    def _texture_analysis(self, image):
        """Texture analysis using statistical methods"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Local binary patterns approximation
        texture = np.zeros_like(gray)

        # Calculate local variance as texture measure
        kernel = np.ones((5, 5)) / 25
        local_mean = ndimage.convolve(gray, kernel)
        local_var = ndimage.convolve(gray**2, kernel) - local_mean**2
        texture = np.sqrt(local_var)

        # Normalize
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)

        return texture

    def _gradient_analysis(self, image):
        """Comprehensive gradient analysis"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Calculate gradients
        grad_y, grad_x = np.gradient(gray)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)

        # Calculate gradient statistics
        grad_stats = {
            'mean_magnitude': np.mean(grad_magnitude),
            'std_magnitude': np.std(grad_magnitude),
            'mean_direction': np.mean(grad_direction),
            'std_direction': np.std(grad_direction)
        }

        return {
            'x': grad_x,
            'y': grad_y,
            'magnitude': grad_magnitude,
            'direction': grad_direction,
            'statistics': grad_stats
        }

    def _frequency_analysis(self, image):
        """Frequency domain analysis"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # 2D FFT
        fft = fftshift(fft2(gray))
        magnitude = np.abs(fft)
        phase = np.angle(fft)

        # Radial frequency analysis
        h, w = gray.shape
        y, x = np.ogrid[:h, :w]
        center = (h//2, w//2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        # Bin frequencies by radius
        r_bins = np.arange(0, min(h, w)//2, 5)
        radial_profile = []

        for i in range(len(r_bins)-1):
            mask = (r >= r_bins[i]) & (r < r_bins[i+1])
            if np.any(mask):
                radial_profile.append(np.mean(magnitude[mask]))
            else:
                radial_profile.append(0)

        return {
            'fft_magnitude': magnitude,
            'fft_phase': phase,
            'radial_profile': np.array(radial_profile),
            'frequency_bins': r_bins[:-1]
        }