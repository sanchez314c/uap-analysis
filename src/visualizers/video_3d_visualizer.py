import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
import cv2

def process_frame_3d(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find the brightest region
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (assumed to be the UAP)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Extract region around the UAP
    roi = gray[max(0, y-20):min(gray.shape[0], y+h+20),
              max(0, x-20):min(gray.shape[1], x+w+20)]
    
    if roi.size == 0:
        return None
        
    return roi

# Set up the output video
frame_dir = "/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/GPT/frames"
output_path = "/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/analysis/3d_visualization.mp4"

frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
sample_frame = cv2.imread(os.path.join(frame_dir, frames[0]))
height, width = sample_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 24.0, (width*2, height))

# Create figure for 3D plot
plt.ion()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

for frame_file in frames:
    # Read frame
    frame = cv2.imread(os.path.join(frame_dir, frame_file))
    
    # Process frame for 3D visualization
    roi = process_frame_3d(frame)
    
    if roi is not None:
        # Clear previous 3D plot
        ax.clear()
        
        # Create mesh grid
        x, y = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
        
        # Plot surface
        surf = ax.plot_surface(x, y, roi, cmap='plasma', alpha=0.8)
        
        # Set labels
        ax.set_title('Light Intensity 3D View')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Intensity')
        
        # Adjust view
        ax.view_init(elev=30, azim=45)
        
        # Convert plot to image
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.resize(plot_img, (width, height))
        
        # Combine original frame and 3D visualization
        combined = np.hstack((frame, plot_img))
        
        # Write frame
        out.write(combined)
    
    plt.close()

out.release()
print("3D visualization video created successfully!")
