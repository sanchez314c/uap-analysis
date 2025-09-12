import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

def process_frame(image_path):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Find the brightest point (assumed to be the UAP)
    y, x = np.unravel_index(np.argmax(img_array), img_array.shape)
    
    # Extract concentric rings around the brightest point
    radius = 20  # Adjust based on UAP size in frame
    y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
    distances = np.sqrt(x_grid**2 + y_grid**2)
    
    # Create mask for different rings
    mask = distances <= radius
    
    # Extract region around brightest point
    region = img_array[
        max(0, y-radius):min(img_array.shape[0], y+radius+1),
        max(0, x-radius):min(img_array.shape[1], x+radius+1)
    ]
    
    return region

# Process frames and create 3D visualization
frame_dir = "/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/analysis/pulse_sequence_frames"
frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

# Create 3D plot
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Process every 10th frame to reduce complexity
for i, frame in enumerate(frames[::10]):
    if i > 50:  # Limit to first 50 samples for visualization
        break
    
    region = process_frame(os.path.join(frame_dir, frame))
    
    # Create mesh grid for 3D surface
    x, y = np.meshgrid(np.arange(region.shape[1]), np.arange(region.shape[0]))
    
    # Plot surface
    ax.plot_surface(x, y, region, cmap='plasma', alpha=0.5)

ax.set_title('UAP Light Intensity 3D Visualization')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Light Intensity')

plt.savefig('/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/analysis/3d_luminance_map.png',
            dpi=300, bbox_inches='tight')
