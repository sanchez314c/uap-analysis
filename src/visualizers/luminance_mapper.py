import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

def analyze_concentric_rings(image_path):
    # Load and convert image to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Find brightest point (UAP center)
    y, x = np.unravel_index(np.argmax(img_array), img_array.shape)
    
    # Define ring radii
    core_radius = 5
    middle_radius = 10
    outer_radius = 15
    
    # Create distance matrix
    y_grid, x_grid = np.ogrid[-outer_radius:outer_radius+1, -outer_radius:outer_radius+1]
    distances = np.sqrt(x_grid**2 + y_grid**2)
    
    # Extract region around UAP
    region = img_array[
        max(0, y-outer_radius):min(img_array.shape[0], y+outer_radius+1),
        max(0, x-outer_radius):min(img_array.shape[1], x+outer_radius+1)
    ]
    
    # Create masks for each ring
    core_mask = distances <= core_radius
    middle_mask = (distances > core_radius) & (distances <= middle_radius)
    outer_mask = (distances > middle_radius) & (distances <= outer_radius)
    
    return region, (core_mask, middle_mask, outer_mask)

# Process frames
frame_dir = "/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/analysis/pulse_sequence_frames"
frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

# Create figure for 3D visualization
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Process key frames
for i, frame in enumerate(frames[::5]):  # Process every 5th frame
    if i > 20:  # Limit to first 20 samples for clarity
        break
        
    region, masks = analyze_concentric_rings(os.path.join(frame_dir, frame))
    
    # Create mesh grid
    x, y = np.meshgrid(np.arange(region.shape[1]), np.arange(region.shape[0]))
    
    # Plot surface with different colors for each ring
    ax.plot_surface(x, y, region * masks[0], color='red', alpha=0.7)    # Core
    ax.plot_surface(x, y, region * masks[1], color='yellow', alpha=0.5) # Middle
    ax.plot_surface(x, y, region * masks[2], color='blue', alpha=0.3)   # Outer

ax.set_title('UAP Light Intensity - Concentric Ring Analysis')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Light Intensity')

# Save the visualization
plt.savefig('/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/analysis/luminance_3d_map.png',
            dpi=300, bbox_inches='tight')

# Print intensity statistics for each ring
print("Analysis complete. 3D visualization saved.")
