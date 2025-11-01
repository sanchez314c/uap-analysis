#!/usr/bin/env python3
"""
3I Atlas Comprehensive Analysis Pipeline
======================================
Complete analysis system for investigating the 3I/ATLAS interstellar object image.
This pipeline combines astronomical analysis, anomaly detection, forensic analysis,
and specialized interstellar object detection methods.

Author: Claude Code Analysis System
Date: October 12, 2025
Version: 1.0.0

Usage:
    python analyze_3i_atlas.py
    python analyze_3i_atlas.py --image /path/to/image.tif
    python analyze_3i_atlas.py --verbose --output results.json
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import traceback

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from processors.image_processor import AtlasImageProcessor
    from analyzers.astronomical_analyzer import AstronomicalAnalyzer
    from analyzers.anomaly_detector import AnomalyDetector
    from analyzers.forensic_analyzer import ForensicAnalyzer
except ImportError as e:
    print(f"Error importing analysis modules: {e}")
    print("Make sure all required modules are in the src/ directory")
    sys.exit(1)

class ThreeIAtlasAnalyzer:
    """Comprehensive analysis pipeline for 3I/ATLAS investigation"""

    def __init__(self, config_file=None):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = self.load_config(config_file)

        # Initialize analyzers
        self.image_processor = AtlasImageProcessor(self.config)
        self.astronomical_analyzer = AstronomicalAnalyzer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.forensic_analyzer = ForensicAnalyzer(self.config)

        self.logger.info("3I Atlas Analysis Pipeline initialized")

    def setup_logging(self, verbose=False):
        """Setup comprehensive logging"""
        log_level = logging.DEBUG if verbose else logging.INFO

        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Setup file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"3i_atlas_analysis_{timestamp}.log"

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            'analysis': {
                'enable_astronomical': True,
                'enable_anomaly_detection': True,
                'enable_forensic': True,
                'enable_interstellar_signature': True
            },
            'processing': {
                'enhance_image': True,
                'extract_features': True,
                'gpu_acceleration': False,
                'use_opencl': False,  # Disable OpenCL for compatibility
                'force_cpu': True  # Force CPU processing for stability
            },
            'output': {
                'save_intermediate_results': True,
                'generate_visualizations': True,
                'create_detailed_report': True
            },
            'thresholds': {
                'anomaly_sensitivity': 0.7,
                'manipulation_threshold': 0.5,
                'interstellar_signature_threshold': 0.6
            }
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                for section, values in user_config.items():
                    if section in default_config:
                        default_config[section].update(values)
                    else:
                        default_config[section] = values
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")

        return default_config

    def analyze_image(self, image_path):
        """Run complete analysis on the 3I/ATLAS image"""
        self.logger.info(f"Starting comprehensive analysis of: {image_path}")

        results = {
            'metadata': {
                'image_path': str(image_path),
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_version': '1.0.0',
                'config': self.config
            },
            'image_info': {},
            'processing_results': {},
            'astronomical_analysis': {},
            'anomaly_detection': {},
            'forensic_analysis': {},
            'interstellar_analysis': {},
            'overall_assessment': {}
        }

        try:
            # Step 1: Load and process image
            self.logger.info("=" * 60)
            self.logger.info("STEP 1: Image Loading and Processing")
            self.logger.info("=" * 60)

            image = self.image_processor.load_tiff(image_path)
            results['image_info'] = {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'min_value': float(np.min(image)),
                'max_value': float(np.max(image)),
                'mean_value': float(np.mean(image))
            }

            # Enhance image if configured
            if self.config['processing']['enhance_image']:
                self.logger.info("Enhancing image...")
                enhanced_image = self.image_processor.enhance_image(image)
                results['processing_results']['enhancement'] = 'completed'
            else:
                enhanced_image = image
                results['processing_results']['enhancement'] = 'skipped'

            # Extract features if configured
            if self.config['processing']['extract_features']:
                self.logger.info("Extracting image features...")
                features = self.image_processor.extract_features(enhanced_image)
                results['processing_results']['features'] = {
                    'edges_detected': True,
                    'texture_analyzed': True,
                    'gradients_computed': True,
                    'frequency_analyzed': True
                }

            # Step 2: Astronomical Analysis
            if self.config['analysis']['enable_astronomical']:
                self.logger.info("=" * 60)
                self.logger.info("STEP 2: Astronomical Analysis")
                self.logger.info("=" * 60)

                self.logger.info("Performing astronomical object analysis...")
                astronomical_results = self.astronomical_analyzer.analyze(enhanced_image)
                results['astronomical_analysis'] = astronomical_results

                self.logger.info(f"Object classification: {astronomical_results.get('object_classification', {}).get('type', 'unknown')}")
                self.logger.info(f"Anomaly score: {astronomical_results.get('anomaly_score', 0):.3f}")

            # Step 3: Anomaly Detection
            if self.config['analysis']['enable_anomaly_detection']:
                self.logger.info("=" * 60)
                self.logger.info("STEP 3: Anomaly Detection")
                self.logger.info("=" * 60)

                self.logger.info("Running comprehensive anomaly detection...")
                anomaly_results = self.anomaly_detector.analyze(enhanced_image)
                results['anomaly_detection'] = anomaly_results

                overall_anomaly_score = anomaly_results.get('overall_anomaly_score', 0)
                self.logger.info(f"Overall anomaly score: {overall_anomaly_score:.3f}")

                # Log significant anomalies
                anomaly_summary = anomaly_results.get('anomaly_summary', {})
                if anomaly_summary.get('critical_anomalies'):
                    self.logger.warning(f"Critical anomalies detected: {len(anomaly_summary['critical_anomalies'])}")
                if anomaly_summary.get('warnings'):
                    self.logger.warning(f"Analysis warnings: {len(anomaly_summary['warnings'])}")

            # Step 4: Forensic Analysis
            if self.config['analysis']['enable_forensic']:
                self.logger.info("=" * 60)
                self.logger.info("STEP 4: Forensic Analysis")
                self.logger.info("=" * 60)

                self.logger.info("Performing forensic authenticity analysis...")
                forensic_results = self.forensic_analyzer.analyze(image_path, enhanced_image)
                results['forensic_analysis'] = forensic_results

                authenticity_score = forensic_results.get('authenticity_score', 0)
                self.logger.info(f"Image authenticity score: {authenticity_score:.3f}")

                forensic_summary = forensic_results.get('forensic_summary', {})
                self.logger.info(f"Authenticity assessment: {forensic_summary.get('authenticity_assessment', 'UNKNOWN')}")
                self.logger.info(f"Risk level: {forensic_summary.get('risk_level', 'UNKNOWN')}")

            # Step 5: Interstellar Object Analysis
            if self.config['analysis']['enable_interstellar_signature']:
                self.logger.info("=" * 60)
                self.logger.info("STEP 5: Interstellar Object Signature Analysis")
                self.logger.info("=" * 60)

                interstellar_results = self._analyze_interstellar_signatures(results)
                results['interstellar_analysis'] = interstellar_results

            # Step 6: Overall Assessment
            self.logger.info("=" * 60)
            self.logger.info("STEP 6: Overall Assessment")
            self.logger.info("=" * 60)

            overall_assessment = self._generate_overall_assessment(results)
            results['overall_assessment'] = overall_assessment

            self.logger.info("ANALYSIS COMPLETE")
            self.logger.info("=" * 60)
            self.logger.info(f"Final Classification: {overall_assessment.get('classification', 'UNKNOWN')}")
            self.logger.info(f"Confidence Level: {overall_assessment.get('confidence', 0):.1%}")
            self.logger.info(f"Anomaly Level: {overall_assessment.get('anomaly_level', 'UNKNOWN')}")
            self.logger.info(f"Recommendation: {overall_assessment.get('recommendation', 'NO RECOMMENDATION')}")

            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            self.logger.error(traceback.format_exc())
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
            return results

    def _analyze_interstellar_signatures(self, results):
        """Specialized analysis for interstellar object signatures"""
        interstellar = {
            'signature_indicators': {},
            'trajectory_analysis': {},
            'composition_analysis': {},
            'behavioral_indicators': {},
            'technological_indicators': {},
            'interstellar_probability': 0.0
        }

        # Analyze astronomical results for interstellar signatures
        astro_results = results.get('astronomical_analysis', {})

        # Check for unusual color indices
        spectral = astro_results.get('spectral_signature', {})
        if 'color_indices' in spectral:
            color_indices = spectral['color_indices']
            b_v_index = color_indices.get('b_v_index', 0)
            # Unusual B-V index for interstellar objects
            if abs(b_v_index - 0.5) > 0.3:
                interstellar['signature_indicators']['unusual_color'] = True
                interstellar['signature_indicators']['color_anomaly_score'] = abs(b_v_index - 0.5)

        # Check for coma or tail structures
        photometric = astro_results.get('photometric_analysis', {})
        if photometric.get('objects_detected', 0) > 0:
            objects = photometric['objects']
            for obj in objects:
                # Look for extended objects with low surface brightness
                if obj['area'] > 100 and obj['mean_brightness'] < 0.5:
                    interstellar['signature_indicators']['extended_structure'] = True
                    break

        # Check morphological features
        morphological = astro_results.get('morphological_analysis', {})
        if morphological.get('objects_analyzed', 0) > 0:
            for obj in morphological.get('morphological_data', []):
                # Unusual shapes could indicate artificial origin
                if obj['circularity'] < 0.1 and obj['solidity'] > 0.9:
                    interstellar['technological_indicators']['geometric_precision'] = True

        # Analyze anomaly detection results
        anomaly_results = results.get('anomaly_detection', {})
        overall_anomaly_score = anomaly_results.get('overall_anomaly_score', 0)

        if overall_anomaly_score > self.config['thresholds']['anomaly_sensitivity']:
            interstellar['signature_indicators']['high_anomaly_score'] = overall_anomaly_score

        # Check forensic results for authenticity
        forensic_results = results.get('forensic_analysis', {})
        authenticity_score = forensic_results.get('authenticity_score', 0)

        if authenticity_score > 0.8:
            interstellar['signature_indicators']['authentic_image'] = True

        # Calculate overall interstellar probability
        probability = 0.0
        indicators_count = 0

        for indicator, value in interstellar['signature_indicators'].items():
            if isinstance(value, bool) and value:
                probability += 0.3
                indicators_count += 1
            elif isinstance(value, (int, float)) and value > 0:
                probability += min(value, 1.0) * 0.3
                indicators_count += 1

        for indicator, value in interstellar['technological_indicators'].items():
            if value:
                probability += 0.5
                indicators_count += 1

        if indicators_count > 0:
            interstellar['interstellar_probability'] = min(1.0, probability / indicators_count)

        return interstellar

    def _generate_overall_assessment(self, results):
        """Generate overall assessment of the image"""
        assessment = {
            'classification': 'UNKNOWN',
            'confidence': 0.0,
            'anomaly_level': 'UNKNOWN',
            'recommendation': 'NO RECOMMENDATION',
            'key_findings': [],
            'risk_factors': [],
            'scientific_value': 'UNKNOWN'
        }

        # Get results from different analyses
        astro_results = results.get('astronomical_analysis', {})
        anomaly_results = results.get('anomaly_detection', {})
        forensic_results = results.get('forensic_analysis', {})
        interstellar_results = results.get('interstellar_analysis', {})

        # Classification
        astro_classification = astro_results.get('object_classification', {})
        if astro_classification:
            classification_type = astro_classification.get('type', 'unknown')
            classification_confidence = astro_classification.get('confidence', 0)

            assessment['classification'] = classification_type.upper()
            assessment['confidence'] = classification_confidence

        # Anomaly level
        overall_anomaly_score = anomaly_results.get('overall_anomaly_score', 0)
        if overall_anomaly_score > 0.8:
            assessment['anomaly_level'] = 'CRITICAL'
        elif overall_anomaly_score > 0.6:
            assessment['anomaly_level'] = 'HIGH'
        elif overall_anomaly_score > 0.4:
            assessment['anomaly_level'] = 'MODERATE'
        elif overall_anomaly_score > 0.2:
            assessment['anomaly_level'] = 'LOW'
        else:
            assessment['anomaly_level'] = 'MINIMAL'

        # Interstellar probability
        interstellar_prob = interstellar_results.get('interstellar_probability', 0)
        if interstellar_prob > self.config['thresholds']['interstellar_signature_threshold']:
            assessment['classification'] = 'POTENTIAL INTERSTELLAR OBJECT'
            assessment['confidence'] = max(assessment['confidence'], interstellar_prob)

        # Key findings
        if astro_classification.get('reasoning'):
            assessment['key_findings'].extend(astro_classification['reasoning'])

        anomaly_summary = anomaly_results.get('anomaly_summary', {})
        if anomaly_summary.get('critical_anomalies'):
            assessment['key_findings'].append(f"Detected {len(anomaly_summary['critical_anomalies'])} critical anomalies")

        # Risk factors
        forensic_summary = forensic_results.get('forensic_summary', {})
        if forensic_summary.get('risk_level') in ['HIGH', 'CRITICAL']:
            assessment['risk_factors'].append("Low image authenticity")

        if overall_anomaly_score > 0.7:
            assessment['risk_factors'].append("High anomaly score")

        # Recommendation
        if assessment['classification'] == 'POTENTIAL INTERSTELLAR OBJECT':
            assessment['recommendation'] = 'IMMEDIATE SCIENTIFIC INVESTIGATION REQUIRED'
            assessment['scientific_value'] = 'EXTREMELY HIGH'
        elif assessment['anomaly_level'] in ['CRITICAL', 'HIGH']:
            assessment['recommendation'] = 'DETAILED SCIENTIFIC ANALYSIS RECOMMENDED'
            assessment['scientific_value'] = 'HIGH'
        elif assessment['anomaly_level'] == 'MODERATE':
            assessment['recommendation'] = 'FURTHER INVESTIGATION SUGGESTED'
            assessment['scientific_value'] = 'MODERATE'
        else:
            assessment['recommendation'] = 'ROUTINE ANALYSIS COMPLETED'
            assessment['scientific_value'] = 'LOW'

        return assessment

    def save_results(self, results, output_file=None):
        """Save analysis results to file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/3i_atlas_analysis_{timestamp}.json"

        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        results_serializable = self._make_json_serializable(results)

        try:
            with open(output_file, 'w') as f:
                json.dump(results_serializable, f, indent=2, default=str)

            self.logger.info(f"Results saved to: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return None

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="3I Atlas Comprehensive Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_3i_atlas.py
  python analyze_3i_atlas.py --image noirlab2522b.tif
  python analyze_3i_atlas.py --verbose --output results.json
        """
    )

    parser.add_argument(
        '--image', '-i',
        help='Path to the 3I/ATLAS TIFF image file',
        default='noirlab2522b.tif'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output JSON file for results',
        default=None
    )

    parser.add_argument(
        '--config', '-c',
        help='Configuration file path',
        default=None
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        print("Please ensure the TIFF file is in the current directory or provide the correct path")
        return 1

    # Initialize analyzer
    try:
        analyzer = ThreeIAtlasAnalyzer(args.config)
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        return 1

    # Run analysis
    print("Starting 3I Atlas Analysis...")
    print(f"Image: {image_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-" * 60)

    results = analyzer.analyze_image(image_path)

    # Save results
    output_file = analyzer.save_results(results, args.output)

    # Print summary
    print("-" * 60)
    print("ANALYSIS SUMMARY")
    print("-" * 60)

    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return 1

    overall = results.get('overall_assessment', {})
    print(f"Classification: {overall.get('classification', 'UNKNOWN')}")
    print(f"Confidence: {overall.get('confidence', 0):.1%}")
    print(f"Anomaly Level: {overall.get('anomaly_level', 'UNKNOWN')}")
    print(f"Scientific Value: {overall.get('scientific_value', 'UNKNOWN')}")
    print(f"Recommendation: {overall.get('recommendation', 'NO RECOMMENDATION')}")

    if output_file:
        print(f"📄 Detailed results saved to: {output_file}")

    # Print key findings
    key_findings = overall.get('key_findings', [])
    if key_findings:
        print("\nKey Findings:")
        for finding in key_findings:
            print(f"  • {finding}")

    # Print risk factors
    risk_factors = overall.get('risk_factors', [])
    if risk_factors:
        print("\nRisk Factors:")
        for risk in risk_factors:
            print(f"  ⚠️  {risk}")

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    # Import numpy at runtime to avoid import errors if numpy is not available
    try:
        import numpy as np
    except ImportError:
        print("Error: numpy is required. Please install it with: pip install numpy")
        sys.exit(1)

    sys.exit(main())