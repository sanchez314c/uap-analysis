#!/usr/bin/env python3
"""
Quick 3I Atlas Analysis
======================
Streamlined analysis focusing on key findings without exhaustive processing.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from processors.image_processor import AtlasImageProcessor
from analyzers.astronomical_analyzer import AstronomicalAnalyzer
from visualizers.comprehensive_visualizer import ComprehensiveVisualizer

def quick_analysis():
    """Run streamlined analysis on 3I/ATLAS image"""
    print("Starting Quick 3I Atlas Analysis...")

    # Initialize analyzers
    image_processor = AtlasImageProcessor()
    astronomical_analyzer = AstronomicalAnalyzer()
    visualizer = ComprehensiveVisualizer()

    # Load image
    print("Loading image...")
    image_path = "noirlab2522b.tif"
    image = image_processor.load_tiff(image_path)
    print(f"✅ Loaded: {image.shape}, dtype: {image.dtype}")

    # Basic enhancement only
    print("Applying basic enhancement...")
    enhanced = image_processor.enhance_image(image)

    # Key astronomical analyses
    print("\n" + "="*50)
    print("KEY ASTRONOMICAL ANALYSIS")
    print("="*50)

    astro_results = astronomical_analyzer.analyze(enhanced)

    # Extract key findings
    classification = astro_results.get('object_classification', {})
    anomaly_score = astro_results.get('anomaly_score', 0)
    photometric = astro_results.get('photometric_analysis', {})
    spectral = astro_results.get('spectral_signature', {})

    print(f"\n🔬 OBJECT CLASSIFICATION:")
    print(f"   Type: {classification.get('type', 'UNKNOWN').upper()}")
    print(f"   Confidence: {classification.get('confidence', 0):.1%}")
    print(f"   Reasoning: {', '.join(classification.get('reasoning', ['No reasoning provided']))}")

    print(f"\n📊 PHOTOMETRIC ANALYSIS:")
    print(f"   Objects detected: {photometric.get('objects_detected', 0)}")
    if photometric.get('objects_detected', 0) > 0:
        obj = photometric['objects'][0]
        print(f"   Brightest object magnitude: {obj.get('apparent_magnitude', 'N/A'):.2f}")
        print(f"   Peak brightness: {obj.get('peak_brightness', 0):.4f}")
        print(f"   Area: {obj.get('area', 0)} pixels")

    print(f"\n🌈 SPECTRAL ANALYSIS:")
    if 'color_indices' in spectral:
        b_v = spectral['color_indices'].get('b_v_index', 0)
        r_i = spectral['color_indices'].get('r_i_index', 0)
        print(f"   B-V index: {b_v:.3f}")
        print(f"   R-I index: {r_i:.3f}")
        print(f"   Color anomaly: {abs(b_v - 0.5) > 0.3}")

    print(f"\n⚠️  ANOMALY SCORE: {anomaly_score:.3f}")
    if anomaly_score > 0.5:
        print("   ⚠️  HIGH ANOMALY DETECTED")
    elif anomaly_score > 0.2:
        print("   ⚡ MODERATE ANOMALY")
    else:
        print("   ✅ LOW ANOMALY")

    # Special interstellar checks
    print(f"\n🛸 INTERSTELLAR SIGNATURES:")
    interstellar = astro_results.get('interstellar_signature_analysis', {})

    if interstellar.get('asymmetry_index', 0) > 0.3:
        print("   ✓ Asymmetric structure detected")
    if interstellar.get('color_anomaly', False):
        print("   ✓ Unusual color signature")
    if interstellar.get('geometric_anomaly', False):
        print("   ✓ Geometric regularity (potential artificial origin)")

    # Quick forensic check
    print(f"\n🔍 AUTHENTICITY CHECK:")
    # Simple checks - if image looks too perfect or has compression artifacts
    image_std = np.std(enhanced)
    if image_std < 0.1:
        print("   ⚠️  Very low noise - possible manipulation")
    elif image_std > 0.3:
        print("   ✓ Normal noise variation - appears authentic")

    # Overall assessment
    print(f"\n" + "="*50)
    print("OVERALL ASSESSMENT")
    print("="*50)

    if classification.get('type') == 'anomalous' or anomaly_score > 0.4:
        print("🚨 ANOMALOUS OBJECT DETECTED")
        print("   Recommendation: Further expert analysis required")
        print("   Scientific Value: HIGH")
    elif classification.get('type') in ['comet', 'asteroid']:
        print(f"📌 {classification.get('type').upper()} DETECTED")
        print("   Appears to be a natural solar system object")
        print("   Scientific Value: MODERATE")
    else:
        print("❓ UNCERTAIN CLASSIFICATION")
        print("   Requires additional analysis")

    print(f"\n🎨 Generating comprehensive visualizations...")

    # Prepare analysis results for visualization
    analysis_results = {
        'astronomical_analysis': astro_results,
        'processing_results': {
            'features': 'extracted'
        }
    }

    # Generate all visualizations
    visual_outputs = visualizer.generate_all_visualizations(image, enhanced, analysis_results)

    # Create HTML report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_report = visualizer.create_visualization_report(visual_outputs, timestamp)

    print(f"✅ Generated {sum(len(v) for v in visual_outputs.values())} visualization files")
    print(f"📄 HTML report: results/{html_report}")

    # Save quick results
    results = {
        'quick_analysis': {
            'classification': classification,
            'anomaly_score': anomaly_score,
            'interstellar_indicators': interstellar,
            'timestamp': datetime.now().isoformat(),
            'assessment': 'ANOMALOUS' if anomaly_score > 0.4 else 'NATURAL',
            'visualizations': visual_outputs
        }
    }

    output_file = f"results/quick_analysis_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"📄 Quick results saved to: {output_file}")

    print("\n🎯 VISUALIZATION SUMMARY:")
    print("="*50)
    for category, files in visual_outputs.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for name, filename in files.items():
            print(f"  • {filename}")

    print("\n✅ Complete analysis with visualizations done!")

    return results

if __name__ == "__main__":
    quick_analysis()