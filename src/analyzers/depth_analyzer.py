#!/usr/bin/env python3
"""
3D Depth Analysis Script for UAP Video
Created by Cortana for Jason
Date: January 25, 2024
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from midas.model_loader import default_models, load_model

class DepthAnalyzer:
    def __init__(self):
        # Initialize MiDaS model for depth estimation
        self.model_type = "DPT_Large"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = load_model(self.model_type, self.device)
        
        # Initialize video processing parameters
        self.input_size = (384, 384)
        self.output_size = (1920, 1080)
        
    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_CUBIC)
        img = torch.from_numpy(img).float().to(self.device)
        img = img.permute(2, 0, 1).unsqueeze(0) / 255.0
        return img
        
    def postprocess(self, depth_map):
        depth_map = depth_map.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, self.output_size, interpolation=cv2.INTER_CUBIC)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
        return depth_map
        
    def analyze_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, self.output_size)
        
        with torch.no_grad():
            for _ in tqdm(range(total_frames), desc="Processing frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                preprocessed = self.preprocess(frame)
                depth = self.midas(preprocessed)
                depth_colored = self.postprocess(depth)
                
                # Add original frame as overlay
                overlay = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)
                
                # Add depth scale bar
                self.add_depth_scale(overlay)
                
                # Write frame
                out.write(overlay)
                
        cap.release()
        out.release()
        
    def add_depth_scale(self, frame):
        height, width = frame.shape[:2]
        scale_height = 30
        scale_width = width // 4
        
        # Create gradient
        gradient = np.linspace(0, 255, scale_width, dtype=np.uint8)
        gradient = np.tile(gradient, (scale_height, 1))
        gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        
        # Add scale to frame
        y_pos = height - scale_height - 10
        x_pos = (width - scale_width) // 2
        frame[y_pos:y_pos+scale_height, x_pos:x_pos+scale_width] = gradient_colored
        
        # Add text
        cv2.putText(frame, "Near", (x_pos-50, y_pos+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "Far", (x_pos+scale_width+10, y_pos+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def main():
    # Set up paths
    base_dir = Path("/Users/heathen-admin/Desktop/Cortana/Projects/UAP_Videos/IMG_2679")
    input_video = base_dir / "original.mp4"
    output_video = base_dir / "GPT/analysis/3d_depth_analysis.mp4"
    
    # Create and run analyzer
    analyzer = DepthAnalyzer()
    analyzer.analyze_video(str(input_video), str(output_video))
    
if __name__ == "__main__":
    main()
