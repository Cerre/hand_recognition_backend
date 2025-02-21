import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both implementations
from tools.xgboost_predictor import xgboost_method
from hand_recognition import HandDetector
from app.main import count_fingers

def main():
    # Initialize camera and detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    print("\nControls:")
    print("  - ESC or Q: Quit")
    print("\nComparing implementations:")
    print("  Visualization (Blue) vs Main.py (Green)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        frame, hands_data = detector.find_hands(frame)
        
        # Process each hand with both implementations
        for hand_data in hands_data:
            landmarks = hand_data['landmarks']
            handedness = hand_data['handedness']
            
            # Implementation from visualization (using xgboost_method directly)
            vis_count = 0
            finger_indices = [
                (4, 3, 2, 1),    # Thumb
                (8, 7, 6, 5),    # Index
                (12, 11, 10, 9), # Middle
                (16, 15, 14, 13),# Ring
                (20, 19, 18, 17) # Pinky
            ]
            for tip_idx, dip_idx, pip_idx, mcp_idx in finger_indices:
                if xgboost_method(landmarks, tip_idx, dip_idx, pip_idx, mcp_idx):
                    vis_count += 1
            
            # Implementation from main.py
            main_count = count_fingers(landmarks)
            
            # Draw results
            h, w, _ = frame.shape
            y_pos = 30 if handedness == 'Left' else 60
            
            # Draw visualization count in blue
            cv2.putText(
                frame,
                f"{handedness} Hand - Vis: {vis_count}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),  # Blue
                2
            )
            
            # Draw main.py count in green
            cv2.putText(
                frame,
                f"{handedness} Hand - Main: {main_count}",
                (300, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),  # Green
                2
            )
            
            # Draw warning if counts don't match
            if vis_count != main_count:
                cv2.putText(
                    frame,
                    "WARNING: Counts don't match!",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),  # Red
                    2
                )
        
        cv2.imshow('XGBoost Implementation Comparison', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 