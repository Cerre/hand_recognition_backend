import cv2
import numpy as np
import sys
import os
from typing import List, Dict, Callable

# Add parent directory to path to import hand_recognition
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_recognition.detector import HandDetector
from hand_recognition.gesture import GestureAnalyzer

def angle_based_method(points: np.ndarray, tip_idx: int, dip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    """Current method: using angles between joints."""
    analyzer = GestureAnalyzer()
    # Get the three sections of the finger
    tip_to_dip = points[tip_idx] - points[dip_idx]
    dip_to_pip = points[dip_idx] - points[pip_idx]
    pip_to_mcp = points[pip_idx] - points[mcp_idx]
    
    # Calculate angles between sections
    angle1 = analyzer.get_angle_between_vectors(tip_to_dip, dip_to_pip)
    angle2 = analyzer.get_angle_between_vectors(dip_to_pip, pip_to_mcp)
    
    return angle1 < 35 and angle2 < 35

def distance_based_method(points: np.ndarray, tip_idx: int, dip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    """Alternative method: using vertical distance from tip to base."""
    tip_y = points[tip_idx][1]
    mcp_y = points[mcp_idx][1]
    return (tip_y - mcp_y) > 0.2  # Finger is extended if tip is significantly above base

def palm_distance_method(points: np.ndarray, tip_idx: int, dip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    """Method from main backend: using distances from palm center."""
    # Calculate palm center using wrist and MCP joints
    palm_points = points[[0, 5, 9, 13, 17]]  # Wrist and all MCP joints
    palm_center = np.mean(palm_points, axis=0)
    
    # Get reference length (palm width) for normalization
    palm_width = np.linalg.norm(points[5] - points[17])  # Distance between index and pinky MCP
    
    # Get tip distance from palm center
    tip_distance = np.linalg.norm(points[tip_idx] - palm_center)
    
    # Get base distance from palm center
    base_distance = np.linalg.norm(points[mcp_idx] - palm_center)
    
    # Store palm center for visualization
    global _last_palm_center
    _last_palm_center = palm_center
    
    # Finger is extended if tip is significantly further from palm center than base
    return (tip_distance - base_distance) / palm_width > 0.3

# Global variable to store palm center for visualization
_last_palm_center = None

# Dictionary of methods to compare
DETECTION_METHODS = {
    'angle_based': angle_based_method,
    'distance_based': distance_based_method,
    'palm_distance': palm_distance_method
}

def draw_finger_status(frame: np.ndarray, hand_data: Dict, method: Callable, landmarks: np.ndarray, 
                      position: tuple, color: tuple) -> None:
    """Draw finger detection results for a specific method."""
    h, w, _ = frame.shape
    
    def draw_point(point: np.ndarray, label: str = None):
        """Helper to draw a point with optional label."""
        px = (point[:2] * [w, h]).astype(int)
        cv2.circle(frame, px, 4, color, -1)  # Filled circle
        cv2.circle(frame, px, 4, (255, 255, 255), 1)  # White border
        if label:
            cv2.putText(
                frame,
                label,
                (px[0] + 5, px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
        return px

    def draw_line(p1: np.ndarray, p2: np.ndarray):
        """Helper to draw a line between two points."""
        px1 = (p1[:2] * [w, h]).astype(int)
        px2 = (p2[:2] * [w, h]).astype(int)
        cv2.line(frame, px1, px2, color, 1)
    
    finger_indices = [
        ('Thumb', (4, 3, 2, 1)),
        ('Index', (8, 7, 6, 5)),
        ('Middle', (12, 11, 10, 9)),
        ('Ring', (16, 15, 14, 13)),
        ('Pinky', (20, 19, 18, 17))
    ]
    
    # Draw method-specific visualizations
    if method == palm_distance_method and _last_palm_center is not None:
        # Draw palm center
        palm_center_px = draw_point(_last_palm_center, "Palm Center")
        
        # Draw palm width reference
        draw_line(landmarks[5], landmarks[17])
        draw_point(landmarks[5], "Palm Width")
        draw_point(landmarks[17])
        
    elif method == distance_based_method:
        # Draw horizontal reference line at wrist level
        wrist_y = landmarks[0][1]
        cv2.line(frame, (0, int(wrist_y * h)), (w, int(wrist_y * h)), color, 1)
        
    elif method == angle_based_method:
        # Will draw angles during finger processing
        pass
    
    y_offset = 0
    for finger_name, (tip_idx, dip_idx, pip_idx, mcp_idx) in finger_indices:
        is_extended = method(landmarks, tip_idx, dip_idx, pip_idx, mcp_idx)
        status = "Extended" if is_extended else "Closed"
        
        # Draw measurement points and lines specific to each method
        if method == palm_distance_method and _last_palm_center is not None:
            # Draw lines from palm center to tip and base
            tip_px = draw_point(landmarks[tip_idx], "Tip")
            base_px = draw_point(landmarks[mcp_idx], "Base")
            palm_center_px = (_last_palm_center[:2] * [w, h]).astype(int)
            cv2.line(frame, palm_center_px, tip_px, color, 1)
            cv2.line(frame, palm_center_px, base_px, color, 1)
            
        elif method == distance_based_method:
            # Draw vertical distance measurement
            tip_px = draw_point(landmarks[tip_idx], "Tip")
            base_px = draw_point(landmarks[mcp_idx], "Base")
            cv2.line(frame, (tip_px[0], tip_px[1]), (tip_px[0], base_px[1]), color, 1)
            
        elif method == angle_based_method:
            # Draw angle measurement points and lines
            tip_px = draw_point(landmarks[tip_idx], "Tip")
            dip_px = draw_point(landmarks[dip_idx], "DIP")
            pip_px = draw_point(landmarks[pip_idx], "PIP")
            mcp_px = draw_point(landmarks[mcp_idx], "MCP")
            cv2.line(frame, tip_px, dip_px, color, 1)
            cv2.line(frame, dip_px, pip_px, color, 1)
            cv2.line(frame, pip_px, mcp_px, color, 1)
        
        # Draw status text
        cv2.putText(
            frame,
            f"{finger_name}: {status}",
            (position[0], position[1] + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        y_offset += 20

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Initialize detector and analyzer
    detector = HandDetector()
    
    # Window setup
    window_name = 'Finger Detection Methods Comparison'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("\nControls:")
    print("  - ESC or Q: Quit")
    print("  - S: Save current frame")
    print("\nComparing methods:")
    for idx, method_name in enumerate(DETECTION_METHODS.keys()):
        print(f"  {idx + 1}. {method_name}")

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Detect hands
        frame, hands_data = detector.find_hands(frame, draw=True)

        # For each detected hand, show results from all methods
        for hand_idx, hand_data in enumerate(hands_data):
            landmarks = hand_data['landmarks']
            
            # Draw results for each method with different colors
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
            x_positions = [10, 200, 400]  # Different x positions for each method
            
            for (method_name, method), color, x_pos in zip(DETECTION_METHODS.items(), colors, x_positions):
                # Draw method name
                cv2.putText(
                    frame,
                    f"Method: {method_name}",
                    (x_pos, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                # Draw finger statuses
                draw_finger_status(frame, hand_data, method, landmarks, (x_pos, 60), color)

        cv2.imshow(window_name, frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            running = False
        elif key == ord('s'):  # Save frame
            timestamp = cv2.getTickCount()
            filename = f"debug_frames/finger_detection_comparison_{timestamp}.jpg"
            os.makedirs('debug_frames', exist_ok=True)
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")
        
        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 