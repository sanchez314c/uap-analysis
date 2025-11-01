#!/usr/bin/env python3
"""
Basic Test Script for 3I Atlas Analysis
====================================
Quick test to verify core functionality before running full analysis.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

def test_basic_functionality():
    """Test basic image loading and processing"""
    print("Testing basic functionality...")

    # Test image loading
    image_path = Path("noirlab2522b.tif")
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False

    print(f"✅ Found image: {image_path}")

    # Load image
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print("❌ Failed to load image")
            return False

        print(f"✅ Loaded image: shape={image.shape}, dtype={image.dtype}")

        # Test data type conversion
        if image.dtype == np.uint16:
            image_float = image.astype(np.float64) / 65535.0
            image_uint8 = (image_float * 255).astype(np.uint8)
            print("✅ Data type conversion successful")
        else:
            print("⚠️  Unexpected image dtype, but continuing...")

        # Test basic OpenCV operations
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        print(f"✅ OpenCV operations successful: edges shape={edges.shape}")

        # Test numpy operations
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        print(f"✅ Numpy operations successful: mean={mean_val:.2f}, std={std_val:.2f}")

        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        import numpy as np
        print("✅ numpy imported successfully")

        import cv2
        print("✅ opencv-python imported successfully")

        import scipy
        print("✅ scipy imported successfully")

        import matplotlib
        print("✅ matplotlib imported successfully")

        from sklearn.cluster import DBSCAN
        print("✅ sklearn imported successfully")

        # Test custom modules
        sys.path.insert(0, str(Path(__file__).parent / "src"))

        from processors.image_processor import AtlasImageProcessor
        print("✅ AtlasImageProcessor imported successfully")

        from analyzers.astronomical_analyzer import AstronomicalAnalyzer
        print("✅ AstronomicalAnalyzer imported successfully")

        from analyzers.anomaly_detector import AnomalyDetector
        print("✅ AnomalyDetector imported successfully")

        from analyzers.forensic_analyzer import ForensicAnalyzer
        print("✅ ForensicAnalyzer imported successfully")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("3I ATLAS ANALYSIS - BASIC FUNCTIONALITY TEST")
    print("=" * 60)

    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        sys.exit(1)

    print()

    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Ready for full analysis")
    print("=" * 60)
    print("\nYou can now run: python analyze_3i_atlas.py")