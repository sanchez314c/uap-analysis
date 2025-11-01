#!/usr/bin/env python3
"""
Comprehensive Visualizer for 3I Atlas Analysis
=============================================
Generates all the visualization outputs similar to the UAP Analysis Suite,
showing different analysis layers, enhancements, and detection results.

Features:
- Multiple image enhancement views
- Edge detection and feature visualization
- Spectral and color analysis displays
- Anomaly detection heatmaps
- Astronomical measurement overlays
- Comparison views (original vs enhanced)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import logging
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)

class ComprehensiveVisualizer:
    """Generate comprehensive visualization outputs for 3I Atlas analysis"""

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_all_visualizations(self, original_image, enhanced_image, analysis_results):
        """Generate all visualization outputs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        outputs = {
            'enhancement_views': self._create_enhancement_views(original_image, enhanced_image, timestamp),
            'analysis_overlays': self._create_analysis_overlays(enhanced_image, analysis_results, timestamp),
            'spectral_visualizations': self._create_spectral_visualizations(enhanced_image, analysis_results, timestamp),
            'anomaly_visualizations': self._create_anomaly_visualizations(enhanced_image, analysis_results, timestamp),
            'comparison_views': self._create_comparison_views(original_image, enhanced_image, analysis_results, timestamp),
            'astronomical_measurements': self._create_astronomical_measurements(enhanced_image, analysis_results, timestamp)
        }

        self.logger.info(f"Generated {sum(len(v) for v in outputs.values())} visualization files")
        return outputs

    def _create_enhancement_views(self, original, enhanced, timestamp):
        """Create different enhancement views"""
        views = {}

        # Convert to uint8 for display
        if original.dtype == np.float64:
            orig_display = (original * 255).astype(np.uint8)
            enh_display = (enhanced * 255).astype(np.uint8)
        else:
            orig_display = original.astype(np.uint8)
            enh_display = enhanced.astype(np.uint8)

        # 1. Original vs Enhanced comparison
        combined = np.hstack([orig_display, enh_display])
        cv2.imwrite(str(self.output_dir / f"01_enhancement_comparison_{timestamp}.png"), combined)
        views['enhancement_comparison'] = f"01_enhancement_comparison_{timestamp}.png"

        # 2. Multi-scale edge detection
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enh_display, cv2.COLOR_RGB2GRAY)
        else:
            gray = enhanced_display

        edges_canny = cv2.Canny(gray, 50, 150)
        edges_laplacian = cv2.Laplacian(gray, cv2.CV_8U)
        edges_sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3)

        edge_combined = np.vstack([
            np.hstack([gray, edges_canny]),
            np.hstack([edges_laplacian, edges_sobel])
        ])
        cv2.imwrite(str(self.output_dir / f"02_edge_analysis_{timestamp}.png"), edge_combined)
        views['edge_analysis'] = f"02_edge_analysis_{timestamp}.png"

        # 3. Gradient visualization
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

        # Create RGB gradient visualization
        grad_rgb = np.zeros_like(enh_display)
        grad_rgb[:, :, 0] = np.abs(grad_x).astype(np.uint8)
        grad_rgb[:, :, 1] = np.abs(grad_y).astype(np.uint8)
        grad_rgb[:, :, 2] = gradient_magnitude

        cv2.imwrite(str(self.output_dir / f"03_gradient_visualization_{timestamp}.png"), grad_rgb)
        views['gradient_visualization'] = f"03_gradient_visualization_{timestamp}.png"

        return views

    def _create_analysis_overlays(self, image, analysis_results, timestamp):
        """Create analysis overlays showing detected features"""
        overlays = {}

        # Convert to uint8 for display
        if image.dtype == np.float64:
            display_image = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
        else:
            display_image = image.astype(np.uint8)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = display_image

        # 1. Object detection overlay
        astro_results = analysis_results.get('astronomical_analysis', {})
        photometric = astro_results.get('photometric_analysis', {})

        if photometric.get('objects_detected', 0) > 0:
            objects = photometric['objects']
            overlay = display_image.copy()

            for obj in objects[:10]:  # Show top 10 objects
                # Draw bounding box (simplified - just a marker for brightest point)
                # In real implementation, we'd use actual object coordinates
                cv2.circle(overlay, (100, 100), 20, (0, 255, 0), 2)
                cv2.putText(overlay, f"M:{obj.get('apparent_magnitude', 0):.1f}",
                           (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imwrite(str(self.output_dir / f"04_object_detection_{timestamp}.png"), overlay)
            overlays['object_detection'] = f"04_object_detection_{timestamp}.png"

        # 2. Feature extraction overlay
        features = analysis_results.get('processing_results', {}).get('features', {})
        if features:
            overlay = display_image.copy()

            # Edge overlay
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            overlay = cv2.addWeighted(overlay, 0.7, edges_colored, 0.3, 0)

            cv2.imwrite(str(self.output_dir / f"05_feature_overlay_{timestamp}.png"), overlay)
            overlays['feature_overlay'] = f"05_feature_overlay_{timestamp}.png"

        return overlays

    def _create_spectral_visualizations(self, image, analysis_results, timestamp):
        """Create spectral and color analysis visualizations"""
        visualizations = {}

        if len(image.shape) != 3:
            return visualizations

        # Convert to uint8 for display
        if image.dtype == np.float64:
            display_image = (image * 255).astype(np.uint8)
        else:
            display_image = image.astype(np.uint8)

        # 1. Color channel analysis
        r, g, b = cv2.split(display_image)

        # Create color channel visualization
        color_channels = np.vstack([
            np.hstack([r, g, b]),
            np.hstack([
                np.zeros_like(r) + 100,  # R channel label
                np.zeros_like(g) + 100,  # G channel label
                np.zeros_like(b) + 100   # B channel label
            ])
        ]).astype(np.uint8)

        cv2.imwrite(str(self.output_dir / f"06_color_channels_{timestamp}.png"), color_channels)
        visualizations['color_channels'] = f"06_color_channels_{timestamp}.png"

        # 2. Color histogram
        plt.figure(figsize=(10, 6))
        colors = ['red', 'green', 'blue']
        channels = [r, g, b]

        for i, (channel, color) in enumerate(zip(channels, colors)):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            plt.plot(hist, color=color, alpha=0.7, label=f'{color.upper()} channel')

        plt.title('Color Channel Histograms')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"07_color_histogram_{timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close()

        visualizations['color_histogram'] = f"07_color_histogram_{timestamp}.png"

        # 3. Color index visualization
        astro_results = analysis_results.get('astronomical_analysis', {})
        spectral = astro_results.get('spectral_signature', {})

        if spectral:
            # Create color index chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # B-V index visualization
            b_v_index = spectral.get('color_indices', {}).get('b_v_index', 0)
            ax1.bar(['B-V Index'], [b_v_index], color='blue', alpha=0.7)
            ax1.set_ylabel('B-V Index Value')
            ax1.set_title(f'B-V Color Index: {b_v_index:.3f}')
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Typical star')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # R-I index visualization
            r_i_index = spectral.get('color_indices', {}).get('r_i_index', 0)
            ax2.bar(['R-I Index'], [r_i_index], color='red', alpha=0.7)
            ax2.set_ylabel('R-I Index Value')
            ax2.set_title(f'R-I Color Index: {r_i_index:.3f}')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / f"08_color_indices_{timestamp}.png", dpi=150, bbox_inches='tight')
            plt.close()

            visualizations['color_indices'] = f"08_color_indices_{timestamp}.png"

        return visualizations

    def _create_anomaly_visualizations(self, image, analysis_results, timestamp):
        """Create anomaly detection visualizations"""
        visualizations = {}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Convert to uint8 for display
        if image.dtype == np.float64:
            display_image = (image * 255).astype(np.uint8)
        else:
            display_image = image.astype(np.uint8)

        # 1. Anomaly heatmap
        anomaly_results = analysis_results.get('anomaly_detection', {})

        # Create statistical anomaly map
        z_scores = np.abs(stats.zscore(gray.flatten()))
        z_score_map = z_scores.reshape(gray.shape)

        # Normalize to 0-255
        anomaly_heatmap = ((z_score_map - z_score_map.min()) /
                          (z_score_map.max() - z_score_map.min()) * 255).astype(np.uint8)

        # Apply colormap
        colormap = plt.get_cmap('hot')
        colored_heatmap = (colormap(anomaly_heatmap / 255.0) * 255).astype(np.uint8)[:, :, :3]

        cv2.imwrite(str(self.output_dir / f"09_anomaly_heatmap_{timestamp}.png"), colored_heatmap)
        visualizations['anomaly_heatmap'] = f"09_anomaly_heatmap_{timestamp}.png"

        # 2. Frequency domain visualization
        fft = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.log1p(np.abs(fft))
        magnitude_normalized = (magnitude / magnitude.max() * 255).astype(np.uint8)

        cv2.imwrite(str(self.output_dir / f"10_frequency_domain_{timestamp}.png"), magnitude_normalized)
        visualizations['frequency_domain'] = f"10_frequency_domain_{timestamp}.png"

        # 3. Combined anomaly view
        combined = np.hstack([
            display_image,
            cv2.applyColorMap(anomaly_heatmap, cv2.COLORMAP_JET),
            cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_HOT)
        ])

        cv2.imwrite(str(self.output_dir / f"11_anomaly_combined_{timestamp}.png"), combined)
        visualizations['anomaly_combined'] = f"11_anomaly_combined_{timestamp}.png"

        return visualizations

    def _create_comparison_views(self, original, enhanced, analysis_results, timestamp):
        """Create side-by-side comparison views"""
        views = {}

        # Convert to uint8 for display
        if original.dtype == np.float64:
            orig_display = (original * 255).astype(np.uint8)
            enh_display = (enhanced * 255).astype(np.uint8)
        else:
            orig_display = original.astype(np.uint8)
            enh_display = enhanced.astype(np.uint8)

        # Create 2x2 comparison grid
        h, w = orig_display.shape[:2]
        comparison = np.zeros((h*2, w*2, 3), dtype=np.uint8)

        # Top-left: Original
        comparison[:h, :w] = orig_display

        # Top-right: Enhanced
        comparison[:h, w:] = enh_display

        # Bottom-left: Difference
        diff = np.abs(orig_display.astype(np.int16) - enh_display.astype(np.int16))
        diff = np.clip(diff * 3, 0, 255).astype(np.uint8)
        comparison[h:, :w] = diff

        # Bottom-right: Edge enhancement comparison
        if len(enh_display.shape) == 3:
            gray_orig = cv2.cvtColor(orig_display, cv2.COLOR_RGB2GRAY)
            gray_enh = cv2.cvtColor(enh_display, cv2.COLOR_RGB2GRAY)
        else:
            gray_orig = orig_display
            gray_enh = enh_display

        edges_orig = cv2.Canny(gray_orig, 50, 150)
        edges_enh = cv2.Canny(gray_enh, 50, 150)
        edges_combined = np.stack([edges_orig, edges_enh, np.zeros_like(edges_orig)], axis=2)
        comparison[h:, w:] = edges_combined

        cv2.imwrite(str(self.output_dir / f"12_comprehensive_comparison_{timestamp}.png"), comparison)
        views['comprehensive_comparison'] = f"12_comprehensive_comparison_{timestamp}.png"

        return views

    def _create_astronomical_measurements(self, image, analysis_results, timestamp):
        """Create astronomical measurement visualizations"""
        measurements = {}

        astro_results = analysis_results.get('astronomical_analysis', {})

        # 1. Photometric profile visualization
        photometric = astro_results.get('photometric_analysis', {})
        if photometric.get('objects_detected', 0) > 0:
            objects = photometric['objects'][:10]  # Top 10 objects

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Magnitude distribution
            magnitudes = [obj.get('apparent_magnitude', 20) for obj in objects if obj.get('apparent_magnitude', 20) < 20]
            ax1.hist(magnitudes, bins=10, color='blue', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Apparent Magnitude')
            ax1.set_ylabel('Count')
            ax1.set_title('Magnitude Distribution')
            ax1.invert_xaxis()  # Brighter stars have lower magnitude
            ax1.grid(True, alpha=0.3)

            # Brightness vs Area scatter
            areas = [obj.get('area', 0) for obj in objects]
            brightness = [obj.get('peak_brightness', 0) for obj in objects]

            ax2.scatter(areas, brightness, alpha=0.7, color='red')
            ax2.set_xlabel('Area (pixels)')
            ax2.set_ylabel('Peak Brightness')
            ax2.set_title('Brightness vs Area')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / f"13_photometric_analysis_{timestamp}.png", dpi=150, bbox_inches='tight')
            plt.close()

            measurements['photometric_analysis'] = f"13_photometric_analysis_{timestamp}.png"

        # 2. Luminosity profile
        luminosity = astro_results.get('luminosity_profile_analysis', {})
        if luminosity.get('radial_profile'):
            fig, ax = plt.subplots(figsize=(8, 6))

            radial_distances = luminosity.get('radial_distances', [])
            radial_profile = luminosity.get('radial_profile', [])

            ax.plot(radial_distances, radial_profile, 'b-', linewidth=2, label='Measured Profile')

            # Fit PSF if available
            if luminosity.get('fitted_psf'):
                fitted_psf = luminosity.get('fitted_psf', [])
                ax.plot(radial_distances, fitted_psf, 'r--', linewidth=2, label='PSF Fit')

            ax.set_xlabel('Radial Distance (pixels)')
            ax.set_ylabel('Luminosity')
            ax.set_title('Radial Luminosity Profile')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / f"14_luminosity_profile_{timestamp}.png", dpi=150, bbox_inches='tight')
            plt.close()

            measurements['luminosity_profile'] = f"14_luminosity_profile_{timestamp}.png"

        # 3. Summary metrics display
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        # Create summary text
        summary_text = f"""
        3I ATLAS ANALYSIS SUMMARY
        =====================

        Classification: {astro_results.get('object_classification', {}).get('type', 'UNKNOWN').upper()}
        Confidence: {astro_results.get('object_classification', {}).get('confidence', 0):.1%}
        Anomaly Score: {astro_results.get('anomaly_score', 0):.3f}

        Objects Detected: {photometric.get('objects_detected', 0)}
        Brightest Magnitude: {photometric.get('objects', [{}])[0].get('apparent_magnitude', 'N/A') if photometric.get('objects') else 'N/A'}

        Color Indices:
        B-V: {astro_results.get('spectral_signature', {}).get('color_indices', {}).get('b_v_index', 0):.3f}
        R-I: {astro_results.get('spectral_signature', {}).get('color_indices', {}).get('r_i_index', 0):.3f}

        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        ax.text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace',
                verticalalignment='center', transform=ax.transAxes)

        plt.savefig(self.output_dir / f"15_analysis_summary_{timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close()

        measurements['analysis_summary'] = f"15_analysis_summary_{timestamp}.png"

        return measurements

    def create_visualization_report(self, all_outputs, timestamp):
        """Create an HTML report with all visualizations"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>3I Atlas Analysis Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; text-align: center; border-bottom: 3px solid #007acc; padding-bottom: 10px; }}
                h2 {{ color: #007acc; margin-top: 30px; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
                .image-item {{ background: #f8f8f8; padding: 15px; border-radius: 8px; text-align: center; }}
                .image-item img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .image-item h3 {{ margin: 10px 0 5px 0; color: #555; }}
                .summary {{ background: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>3I Atlas Analysis Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <div class="summary">
                    <h2>Analysis Summary</h2>
                    <p>Comprehensive astronomical and forensic analysis of the 3I/ATLAS interstellar object image.</p>
                </div>

                <h2>Enhancement Views</h2>
                <div class="image-grid">
        """

        # Add enhancement views
        for name, filename in all_outputs.get('enhancement_views', {}).items():
            html_content += f"""
                    <div class="image-item">
                        <h3>{name.replace('_', ' ').title()}</h3>
                        <img src="{filename}" alt="{name}">
                    </div>
            """

        # Add other sections
        sections = ['analysis_overlays', 'spectral_visualizations', 'anomaly_visualizations',
                    'comparison_views', 'astronomical_measurements']

        for section in sections:
            if section in all_outputs:
                html_content += f"""
                </div>

                <h2>{section.replace('_', ' ').title()}</h2>
                <div class="image-grid">
                """
                for name, filename in all_outputs[section].items():
                    html_content += f"""
                    <div class="image-item">
                        <h3>{name.replace('_', ' ').title()}</h3>
                        <img src="{filename}" alt="{name}">
                    </div>
                    """

        html_content += """
                </div>
            </div>
        </body>
        </html>
        """

        with open(self.output_dir / f"analysis_report_{timestamp}.html", 'w') as f:
            f.write(html_content)

        return f"analysis_report_{timestamp}.html"