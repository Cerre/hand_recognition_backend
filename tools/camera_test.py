import cv2
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path to import hand_recognition
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_recognition.detector import HandDetector
from hand_recognition.gesture import GestureAnalyzer

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Create debug_frames directory if it doesn't exist
    debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_frames')
    os.makedirs(debug_dir, exist_ok=True)

    # Initialize hand detection and gesture analysis
    detector = HandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    analyzer = GestureAnalyzer()

    # Window setup with a larger default size
    window_name = 'Hand Recognition Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # Set default size to 720p
    
    print("Camera test started. Press 'q' to quit.")
    print("Controls:")
    print("  - ESC or Q: Quit")
    print(f"  - S: Save frame to {debug_dir}")
    print("\nHand roles:")
    print("  Left hand: Player number")
    print("  Right hand: Points")

    running = True
    while running:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        # Detect hands and get landmarks
        frame, hands_data = detector.find_hands(frame, draw=True)

        # Analyze each detected hand
        for hand_data in hands_data:
            results = analyzer.analyze_hand(hand_data)
            
            # Get pixel coordinates for text placement
            landmarks = hand_data['landmarks']
            wrist_pos = detector.get_landmark_coords(frame, landmarks)[0]
            
            # Draw results with role
            text = f"{results['role'].title()}: {results['finger_count']}"
            cv2.putText(
                frame,
                text,
                (wrist_pos[0], wrist_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Show frame
        cv2.imshow(window_name, frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            running = False
        elif key == ord('s'):  # Save frame
            # Generate filename with timestamp and hand information
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hand_info = f"{len(hands_data)}hands"
            filename = f"frame_{timestamp}_{hand_info}.jpg"
            filepath = os.path.join(debug_dir, filename)
            
            # Save frame
            cv2.imwrite(filepath, frame)
            print(f"Saved frame to {filepath}")
        
        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 