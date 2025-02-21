import cv2
import numpy as np
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import hand_recognition
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_recognition.detector import HandDetector

def create_directory_structure():
    """Create the directory structure for storing data."""
    base_dir = Path("data_output")
    for person in ["person1", "person2"]:  # Added person2
        for hand in ["left", "right"]:
            for number in range(0, 6):  # 0 to 5 fingers
                number_dir = base_dir / person / hand / str(number)
                (number_dir / "images").mkdir(parents=True, exist_ok=True)
                (number_dir / "landmarks").mkdir(parents=True, exist_ok=True)
    return base_dir

def save_data(frame, hands_data, number, base_dir, person_id):
    """Save the image and landmark data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Save each detected hand separately
    for idx, hand_data in enumerate(hands_data):
        handedness = hand_data.get('handedness', 'unknown')
        
        # Create filename with hand information
        filename_base = f"{handedness}_{timestamp}_{idx}"
        
        # Save image
        image_path = base_dir / person_id / handedness.lower() / str(number) / "images" / f"{filename_base}.jpg"
        cv2.imwrite(str(image_path), frame)
        
        # Save landmarks
        landmark_path = base_dir / person_id / handedness.lower() / str(number) / "landmarks" / f"{filename_base}.json"
        with open(landmark_path, 'w') as f:
            json.dump({
                'handedness': handedness,
                'landmarks': hand_data['landmarks'].tolist(),
                'number': number,
                'timestamp': timestamp,
                'person_id': person_id
            }, f, indent=2)
        
        print(f"Saved {handedness} hand showing {number} fingers - {filename_base}")

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Initialize detector
    detector = HandDetector()
    
    # Create directory structure
    base_dir = create_directory_structure()
    
    # Window setup
    window_name = 'Hand Gesture Data Collector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Current number being collected and person ID
    current_number = None
    person_id = "person1"  # Set to person1
    
    print("\nControls:")
    print("  - 0-5: Select number of fingers to collect")
    print("  - S: Save current frame and landmarks")
    print("  - ESC or Q: Quit")
    print("\nCollecting data for:", person_id)
    print("\nInstructions:")
    print("1. Press a number (0-5) to start collecting that gesture")
    print("2. Show the corresponding number of fingers")
    print("3. Press 'S' to save the frame and landmarks")
    print("4. Repeat with different hand positions")
    print("5. Press another number to switch or ESC/Q to quit\n")

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Detect hands
        frame, hands_data = detector.find_hands(frame, draw=True)

        # Draw current number being collected and person ID
        cv2.putText(
            frame,
            f"Person: {person_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        if current_number is not None:
            cv2.putText(
                frame,
                f"Collecting number: {current_number}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Show detected hands count and handedness
        y_pos = 110
        cv2.putText(
            frame,
            f"Detected hands: {len(hands_data)}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Show handedness for each detected hand
        for idx, hand_data in enumerate(hands_data):
            y_pos += 40
            handedness = hand_data.get('handedness', 'unknown')
            cv2.putText(
                frame,
                f"Hand {idx + 1}: {handedness}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow(window_name, frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Number selection
        if ord('0') <= key <= ord('5'):
            current_number = key - ord('0')
            print(f"\nNow collecting gesture for number: {current_number}")
        
        # Save frame
        elif key == ord('s') and current_number is not None and hands_data:
            save_data(frame, hands_data, current_number, base_dir, person_id)
        
        # Quit
        elif key == ord('q') or key == 27:  # q or ESC
            running = False
        
        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 