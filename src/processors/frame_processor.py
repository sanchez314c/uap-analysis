import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_frames(frames_path):
    frames = []
    luminosity_values = []
    motion_values = []
    prev_frame = None
    
    # Sort frames numerically
    frame_files = sorted(Path(frames_path).glob('frame_*.png'), 
                        key=lambda x: int(x.stem.split('_')[1]))
    
    for frame_path in frame_files:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
            
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average luminosity
        avg_luminosity = np.mean(gray)
        luminosity_values.append(avg_luminosity)
        
        # Calculate motion if we have a previous frame
        if prev_frame is not None:
            motion = cv2.absdiff(prev_frame, gray)
            motion_value = np.mean(motion)
            motion_values.append(motion_value)
        
        prev_frame = gray
        frames.append(frame)
    
    # Add 0 for first frame motion to match array lengths
    motion_values.insert(0, 0)
    
    return np.array(frames), np.array(luminosity_values), np.array(motion_values)

def find_anomalies(values, threshold=2):
    mean = np.mean(values)
    std = np.std(values)
    z_scores = (values - mean) / std
    return np.where(np.abs(z_scores) > threshold)[0]

def plot_analysis(luminosity_values, motion_values):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot luminosity
    ax1.plot(luminosity_values, label='Luminosity')
    ax1.set_title('Luminosity Over Time')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Average Luminosity')
    ax1.grid(True)
    
    # Plot motion
    ax2.plot(motion_values, label='Motion')
    ax2.set_title('Motion Between Frames')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Motion Value')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('analysis_results.png')
    plt.close()

def main():
    frames_path = '.'
    frames, luminosity, motion = analyze_frames(frames_path)
    
    # Find anomalies
    lum_anomalies = find_anomalies(luminosity)
    motion_anomalies = find_anomalies(motion)
    
    # Plot results
    plot_analysis(luminosity, motion)
    
    print(f"Found {len(lum_anomalies)} luminosity anomalies")
    print(f"Found {len(motion_anomalies)} motion anomalies")
    
    # Print frame numbers with anomalies
    print("\nLuminosity anomaly frames:", lum_anomalies)
    print("Motion anomaly frames:", motion_anomalies)

if __name__ == "__main__":
    main()
