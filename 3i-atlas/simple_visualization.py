#!/usr/bin/env python3
"""
Simple 3I Atlas Visualization
============================
Generate basic visualization outputs without complex data processing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def generate_visualizations():
    """Generate basic visualizations of the 3I Atlas image"""
    print("🎨 Generating visualizations...")

    # Load the image
    image = cv2.imread("noirlab2522b.tif", cv2.IMREAD_UNCHANGED)
    if image is None:
        print("❌ Could not load image")
        return

    print(f"✅ Loaded image: {image.shape}, dtype: {image.dtype}")

    # Convert to uint8 if needed
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    visualizations = []

    # 1. Original image
    cv2.imwrite(str(output_dir / f"01_original_{timestamp}.png"), image)
    visualizations.append("01_original.png")

    # 2. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    cv2.imwrite(str(output_dir / f"02_grayscale_{timestamp}.png"), gray)
    visualizations.append("02_grayscale.png")

    # 3. Edge detection
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(str(output_dir / f"03_edges_{timestamp}.png"), edges)
    visualizations.append("03_edges.png")

    # 4. Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(output_dir / f"04_threshold_{timestamp}.png"), thresh)
    visualizations.append("04_threshold.png")

    # 5. Blurred image
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    cv2.imwrite(str(output_dir / f"05_blurred_{timestamp}.png"), blurred)
    visualizations.append("05_blurred.png")

    # 6. Sharpened image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    cv2.imwrite(str(output_dir / f"06_sharpened_{timestamp}.png"), sharpened)
    visualizations.append("06_sharpened.png")

    # 7. Create a 2x2 comparison grid
    h, w = gray.shape
    comparison = np.zeros((h*2, w*2), dtype=np.uint8)
    comparison[:h, :w] = gray
    comparison[:h, w:] = edges
    comparison[h:, :w] = thresh
    comparison[h:, w:] = blurred

    cv2.imwrite(str(output_dir / f"07_comparison_grid_{timestamp}.png"), comparison)
    visualizations.append("07_comparison_grid.png")

    # 8. Color analysis if color image
    if len(image.shape) == 3:
        # Split color channels
        b, g, r = cv2.split(image)

        # Create color channel montage
        color_channels = np.vstack([
            np.hstack([b, g, r]),
            np.hstack([gray, edges, thresh])
        ])
        cv2.imwrite(str(output_dir / f"08_color_analysis_{timestamp}.png"), color_channels)
        visualizations.append("08_color_analysis.png")

    # 9. Histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure(figsize=(10, 6))
    plt.plot(hist, color='blue')
    plt.title('Intensity Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"09_histogram_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()
    visualizations.append("09_histogram.png")

    # 10. Enhanced visualization with contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / f"10_contours_{timestamp}.png"), contour_image)
    visualizations.append("10_contours.png")

    print(f"✅ Generated {len(visualizations)} visualization files:")
    for viz in visualizations:
        print(f"  • {viz}")

    return visualizations

def create_summary_html(visualizations, timestamp):
    """Create a simple HTML summary page"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>3I Atlas Visualization Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            h1 {{ color: #333; text-align: center; }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
            .image-item {{ text-align: center; }}
            .image-item img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>3I Atlas Visualization Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="image-grid">
    """

    for viz in visualizations:
        html_content += f"""
                <div class="image-item">
                    <img src="{viz}" alt="{viz}">
                    <h3>{viz.replace('.png', '').replace('_', ' ').title()}</h3>
                </div>
        """

    html_content += """
            </div>
        </div>
    </body>
    </html>
    """

    output_dir = Path("results")
    with open(output_dir / f"visualization_report_{timestamp}.html", 'w') as f:
        f.write(html_content)

    return f"visualization_report_{timestamp}.html"

if __name__ == "__main__":
    print("="*50)
    print("3I ATLAS VISUALIZATION GENERATOR")
    print("="*50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualizations = generate_visualizations()

    if visualizations:
        html_report = create_summary_html(visualizations, timestamp)
        print(f"\n📄 HTML report: results/{html_report}")
        print(f"\n✅ {len(visualizations)} visualizations created successfully!")
    else:
        print("\n❌ No visualizations created")