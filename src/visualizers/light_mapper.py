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
    
    # Extract region around brightest point
    region = img_array[
        max(0, y-outer_radius):min(img_array.shape[0], y+outer_radius+1),
        max(0, x-outer_radius):min(img_array.shape[1], x+outer_radius+1)
    ]
    
    return region

# Create 3D visualization
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Process frames from the correct directory
frame_dir = "/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/GPT/frames"
frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

# Process key frames
for i, frame in enumerate(frames[::10]):  # Every 10th frame
    if i > 20:  # Limit for clarity
        break
        
    # Get intensity data
    intensity_map = analyze_concentric_rings(os.path.join(frame_dir, frame))
    
    # Create mesh grid
    x, y = np.meshgrid(
        np.arange(intensity_map.shape[1]),
        np.arange(intensity_map.shape[0])
    )
    
    # Plot surface with plasma colormap for better visibility
    ax.plot_surface(
        x, y, intensity_map,
        cmap='plasma',
        alpha=0.7,
        rstride=1,
        cstride=1
    )

ax.set_title('UAP Light Intensity 3D Map')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Light Intensity')

# Adjust view angle for better visualization
ax.view_init(elev=30, azim=45)

plt.savefig('/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/analysis/light_3d_map.png',
            dpi=300,
            bbox_inches='tight')

print("3D Light intensity map created successfully!")
