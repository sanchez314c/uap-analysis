import numpy as np
import pyvista as pv
from PIL import Image
import os
import matplotlib.pyplot as plt
import open3d as o3d

def process_frame_pyvista(image_path):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    intensity = np.array(img, dtype=np.float32) / 255.0
    
    # Create coordinate grids
    y, x = np.mgrid[0:intensity.shape[0], 0:intensity.shape[1]]
    
    # Create points for each pixel
    points = np.column_stack((x.ravel(), y.ravel(), intensity.ravel() * 50))  # Scale Z for visibility
    
    # Create PyVista point cloud
    cloud = pv.PolyData(points)
    cloud['intensity'] = intensity.ravel()
    
    # Create surface from point cloud
    surface = cloud.delaunay_2d()
    
    return surface

def process_frame_open3d(image_path):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    intensity = np.array(img, dtype=np.float32) / 255.0
    
    # Create coordinate grids
    y, x = np.mgrid[0:intensity.shape[0], 0:intensity.shape[1]]
    
    # Create points for Open3D
    points = np.column_stack((x.ravel(), y.ravel(), intensity.ravel() * 50))
    colors = np.column_stack((intensity.ravel(), intensity.ravel(), intensity.ravel()))
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def main():
    # Get frames directory
    frame_dir = "/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/GPT/frames"
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    
    # Process key frames (every 10th frame)
    for i, frame in enumerate(frames[::10]):
        if i > 5:  # Process first 5 key frames for testing
            break
            
        frame_path = os.path.join(frame_dir, frame)
        
        # PyVista visualization
        surface = process_frame_pyvista(frame_path)
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(surface, scalars='intensity', cmap='plasma')
        plotter.camera_position = 'xy'
        plotter.camera.zoom(1.5)
        
        # Save screenshot
        plotter.screenshot(f'/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/analysis/3d_frame_{i:03d}_pyvista.png')
        plotter.close()
        
        # Open3D visualization
        pcd = process_frame_open3d(frame_path)
        
        # Visualize and save
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        
        # Set view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([intensity.shape[1]/2, intensity.shape[0]/2, 0])
        ctr.set_up([0, -1, 0])
        
        # Render and save
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/analysis/3d_frame_{i:03d}_open3d.png')
        vis.destroy_window()

if __name__ == "__main__":
    main()
    print("3D reconstructions completed!")
