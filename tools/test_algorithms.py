import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Add parent directory to path to import hand_recognition
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finger_detection_comparison import (
    angle_based_method,
    adaptive_threshold_method,
    palm_distance_method
)

def load_dataset(base_dir: Path) -> List[Dict]:
    """Load all landmark data from the dataset."""
    dataset = []
    
    # Iterate through all person directories
    for person_dir in base_dir.glob("person*"):
        person_id = person_dir.name
        
        # Iterate through handedness (left/right)
        for hand_dir in person_dir.glob("*"):
            handedness = hand_dir.name
            
            # Iterate through all number directories
            for number_dir in hand_dir.glob("[0-5]"):
                number = int(number_dir.name)
                landmark_dir = number_dir / "landmarks"
                
                # Load all landmark files for this number
                for landmark_file in landmark_dir.glob("*.json"):
                    with open(landmark_file, 'r') as f:
                        data = json.load(f)
                        data.update({
                            'expected_number': number,
                            'person_id': person_id,
                            'handedness': handedness
                        })
                        dataset.append(data)
    
    return dataset

def count_extended_fingers(landmarks: np.ndarray, method) -> int:
    """Count number of extended fingers using the specified method."""
    finger_indices = [
        (4, 3, 2, 1),    # Thumb
        (8, 7, 6, 5),    # Index
        (12, 11, 10, 9), # Middle
        (16, 15, 14, 13),# Ring
        (20, 19, 18, 17) # Pinky
    ]
    
    extended = 0
    for tip_idx, dip_idx, pip_idx, mcp_idx in finger_indices:
        if method(landmarks, tip_idx, dip_idx, pip_idx, mcp_idx):
            extended += 1
    
    return extended

def evaluate_method(dataset: List[Dict], method, method_name: str) -> Dict:
    """Evaluate a method's accuracy on the dataset."""
    results = {
        'total': len(dataset),
        'correct': 0,
        'by_number': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_hand': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_person': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'confusion_matrix': defaultdict(lambda: defaultdict(int))
    }
    
    for data in dataset:
        landmarks = np.array(data['landmarks'])
        expected = data['expected_number']
        handedness = data['handedness']
        person_id = data['person_id']
        
        # Count extended fingers
        predicted = count_extended_fingers(landmarks, method)
        
        # Update statistics
        is_correct = predicted == expected
        results['by_number'][expected]['total'] += 1
        results['by_hand'][handedness]['total'] += 1
        results['by_person'][person_id]['total'] += 1
        results['confusion_matrix'][expected][predicted] += 1
        
        if is_correct:
            results['correct'] += 1
            results['by_number'][expected]['correct'] += 1
            results['by_hand'][handedness]['correct'] += 1
            results['by_person'][person_id]['correct'] += 1
    
    return results

def print_results(results: Dict, method_name: str):
    """Print evaluation results in a readable format."""
    print(f"\n=== Results for {method_name} ===")
    print(f"Overall accuracy: {results['correct'] / results['total']:.2%}")
    
    print("\nAccuracy by number:")
    for number in sorted(results['by_number'].keys()):
        stats = results['by_number'][number]
        accuracy = stats['correct'] / stats['total']
        print(f"  {number}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nAccuracy by hand:")
    for hand in sorted(results['by_hand'].keys()):
        stats = results['by_hand'][hand]
        accuracy = stats['correct'] / stats['total']
        print(f"  {hand}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nAccuracy by person:")
    for person in sorted(results['by_person'].keys()):
        stats = results['by_person'][person]
        accuracy = stats['correct'] / stats['total']
        print(f"  {person}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nConfusion Matrix:")
    print("  Actual → Predicted")
    print("    " + " ".join(f"{i:4}" for i in range(6)))
    for actual in range(6):
        row = [results['confusion_matrix'][actual][pred] for pred in range(6)]
        print(f"  {actual}: " + " ".join(f"{n:4}" for n in row))

def main():
    base_dir = Path("data_output")
    if not base_dir.exists():
        print("Error: No dataset found. Please run data_collector.py first.")
        return
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(base_dir)
    print(f"Loaded {len(dataset)} samples")
    
    # Test all methods
    methods = [
        (angle_based_method, "Angle-based Method"),
        (adaptive_threshold_method, "Adaptive Threshold Method"),
        (palm_distance_method, "Palm Distance Method")
    ]
    
    for method, name in methods:
        results = evaluate_method(dataset, method, name)
        print_results(results, name)

if __name__ == "__main__":
    main() 