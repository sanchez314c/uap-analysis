#!/usr/bin/env python3
"""
3D Reconstruction Script for UAP Video Analysis
Author: Cortana
Date: January 25, 2025
Purpose: Perform 3D reconstruction from video frames using structure from motion
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
import open3d as o3d

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UAP3DReconstructor:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.frames = []
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def extract_frames(self) -> None:
        """Extract frames from video file"""
        cap = cv2.VideoCapture(str(self.video_path))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()
        logger.info(f"Extracted {len(self.frames)} frames")
        
    def detect_features(self, frame: np.ndarray) -> Tuple[List, List]:
        """Detect features in frame using SIFT"""
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(frame, None)
        return keypoints, descriptors
        
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features between two frames"""
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return good_matches
        
    def reconstruct_3d(self) -> o3d.geometry.PointCloud:
        """Perform 3D reconstruction from matched features"""
        self.extract_frames()
        if len(self.frames) < 2:
            raise ValueError("Need at least 2 frames for reconstruction")
            
        # Initialize camera matrix (can be calibrated for better results)
        h, w = self.frames[0].shape[:2]
        self.camera_matrix = np.array([
            [w, 0, w/2],
            [0, w, h/2],
            [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1))
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        points = []
        
        for i in range(len(self.frames)-1):
            kp1, desc1 = self.detect_features(self.frames[i])
            kp2, desc2 = self.detect_features(self.frames[i+1])
            matches = self.match_features(desc1, desc2)
            
            # Get matched point coordinates
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            # Essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix)
            
            # Recover relative camera pose
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
            
            # Triangulate points
            proj1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3,1))))
            proj2 = self.camera_matrix @ np.hstack((R, t))
            
            pts4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
            pts3d = pts4d[:3] / pts4d[3]
            points.extend(pts3d.T)
            
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        return pcd
        
    def save_reconstruction(self, pcd: o3d.geometry.PointCloud, output_path: str):
        """Save the reconstructed point cloud"""
        o3d.io.write_point_cloud(output_path, pcd)
        logger.info(f"Saved reconstruction to {output_path}")

def main():
    video_path = "input_video.mp4"  # Replace with actual video path
    output_path = "reconstruction.ply"
    
    reconstructor = UAP3DReconstructor(video_path)
    try:
        point_cloud = reconstructor.reconstruct_3d()
        reconstructor.save_reconstruction(point_cloud, output_path)
    except Exception as e:
        logger.error(f"Reconstruction failed: {str(e)}")

if __name__ == "__main__":
    main()
