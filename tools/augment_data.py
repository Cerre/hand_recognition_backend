import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import Dict, List
import shutil

def rotate_landmarks(landmarks: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate landmarks around palm center by given angle."""
    # Convert to radians
    angle = np.radians(angle_degrees)
    
    # Calculate palm center
    palm_points = landmarks[[0, 5, 9, 13, 17]]  # Wrist and all MCP joints
    palm_center = np.mean(palm_points, axis=0)
    
    # Create rotation matrix
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Center, rotate, and move back
    centered = landmarks - palm_center
    rotated = np.dot(centered, rot_matrix.T)
    return rotated + palm_center

def flip_landmarks_vertical(landmarks: np.ndarray) -> np.ndarray:
    """Flip landmarks vertically (upside down)."""
    # Calculate palm center
    palm_points = landmarks[[0, 5, 9, 13, 17]]
    palm_center = np.mean(palm_points, axis=0)
    
    # Flip around palm center
    centered = landmarks - palm_center
    centered[:, 1] *= -1  # Flip Y coordinates
    return centered + palm_center

def augment_data(source_dir: Path, target_dir: Path):
    """Augment landmark data with flipped versions."""
    print(f"Augmenting data from {source_dir} to {target_dir}")
    
    # First, copy original data
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    
    # Count files processed and augmentations created
    original_count = 0
    augmented_count = 0
    
    # Process each landmark file
    for landmark_file in source_dir.rglob("*.json"):
        original_count += 1
        
        # Read original data
        with open(landmark_file, 'r') as f:
            data = json.load(f)
        
        # Convert landmarks to numpy array
        landmarks = np.array(data['landmarks'])
        
        # Create augmentations
        augmentations = {
            'flipped': flip_landmarks_vertical(landmarks),
            'flipped_45': flip_landmarks_vertical(rotate_landmarks(landmarks, 45)),
            'flipped_neg45': flip_landmarks_vertical(rotate_landmarks(landmarks, -45))
        }
        
        # Save augmentations
        rel_path = landmark_file.relative_to(source_dir)
        for aug_name, aug_landmarks in augmentations.items():
            # Create augmented filename
            new_path = target_dir / rel_path.parent / f"{landmark_file.stem}_{aug_name}.json"
            
            # Update data with augmented landmarks
            aug_data = data.copy()
            aug_data['landmarks'] = aug_landmarks.tolist()
            aug_data['augmentation'] = aug_name
            
            # Save augmented data
            new_path.parent.mkdir(parents=True, exist_ok=True)
            with open(new_path, 'w') as f:
                json.dump(aug_data, f, indent=2)
            augmented_count += 1
    
    print(f"\nAugmentation complete:")
    print(f"Original files: {original_count}")
    print(f"Augmentations created: {augmented_count}")
    print(f"Total files in target: {original_count + augmented_count}")

def main():
    # Source directory (original data)
    source_dir = Path("data_output/person1")
    
    # Target directory (augmented data - using person3 for augmented data)
    target_dir = Path("ml_data/person3")
    
    # Perform augmentation
    augment_data(source_dir, target_dir)

if __name__ == "__main__":
    main() 