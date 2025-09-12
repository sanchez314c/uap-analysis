import torch
import cv2
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from tqdm import tqdm

class UAP_DimensionalAnalysis:
    def __init__(self):
        # Initialize MiDaS for depth estimation
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_model.to(self.device)

    def extract_depth_map(self, frame):
        # Prepare image for depth estimation
        inputs = self.feature_extractor(images=frame, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Normalize depth map
        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        return depth_map.cpu().numpy()

    def process_luminosity(self, frame):
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_l = clahe.apply(l_channel)
        
        # Calculate luminosity patterns
        luminosity_map = cv2.GaussianBlur(enhanced_l, (5,5), 0)
        return luminosity_map

    def extract_negative_space(self, frame):
        # Edge detection
        edges = cv2.Canny(frame, 100, 200)
        
        # Invert to get negative space
        negative_space = cv2.bitwise_not(edges)
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        negative_space = cv2.morphologyEx(negative_space, cv2.MORPH_CLOSE, kernel)
        
        return negative_space

    def create_3d_2d_inversion(self, frame):
        # Get depth map
        depth_map = self.extract_depth_map(frame)
        
        # Get luminosity
        lum_map = self.process_luminosity(frame)
        
        # Get negative space
        neg_space = self.extract_negative_space(frame)
        
        # Combine all three dimensions
        combined_map = np.stack([depth_map, lum_map, neg_space], axis=-1)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        h, w = combined_map.shape[:2]
        flattened = combined_map.reshape(-1, 3)
        reduced = pca.fit_transform(flattened)
        
        # Reshape back to 2D
        inverted_2d = reduced.reshape(h, w, 2)
        
        return inverted_2d

    def process_video(self, frame_path, total_frames):
        all_inversions = []
        
        for i in tqdm(range(total_frames)):
            frame_file = f"{frame_path}/frame_{str(i).zfill(8)}.png"
            frame = cv2.imread(frame_file)
            
            if frame is None:
                continue
                
            inversion = self.create_3d_2d_inversion(frame)
            all_inversions.append(inversion)
            
        return np.array(all_inversions)

    def visualize_patterns(self, inversions):
        # Create 3D visualization of pattern movement
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot movement patterns
        for t in range(0, len(inversions), 5):  # Sample every 5th frame
            points = inversions[t].reshape(-1, 2)
            hull = ConvexHull(points)
            ax.scatter(points[hull.vertices,0], 
                      points[hull.vertices,1], 
                      t,
                      alpha=0.1)
            
        plt.show()

if __name__ == "__main__":
    analyzer = UAP_DimensionalAnalysis()
    frame_path = "/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/GPT/frames"
    inversions = analyzer.process_video(frame_path, 954)
    analyzer.visualize_patterns(inversions)
    
    # Save the results
    np.save("/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679/GPT/inversions.npy", inversions)
