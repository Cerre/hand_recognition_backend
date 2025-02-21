import cv2
import numpy as np
import sys
import os
from typing import List, Dict, Callable
from xgboost_predictor import xgboost_method  # Add import at the top

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

def adaptive_threshold_method(points: np.ndarray, tip_idx: int, dip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    """Adaptive method: using thresholds that adapt to hand size and orientation."""
    # Calculate hand size (distance between wrist and middle finger MCP)
    hand_size = np.linalg.norm(points[0] - points[9])
    
    # Calculate palm orientation
    palm_direction = points[9] - points[0]  # Middle finger MCP to wrist
    palm_angle = np.arctan2(palm_direction[1], palm_direction[0])
    
    # Adjust thresholds based on hand size and orientation
    angle_threshold = 35 + (20 * abs(np.sin(palm_angle)))  # More lenient when hand is tilted
    distance_threshold = 0.2 * (hand_size / 0.3)  # Scale with hand size
    
    # Get angles
    tip_to_dip = points[tip_idx] - points[dip_idx]
    dip_to_pip = points[dip_idx] - points[pip_idx]
    angle = GestureAnalyzer().get_angle_between_vectors(tip_to_dip, dip_to_pip)
    
    # Get normalized vertical distance
    vertical_dist = (points[tip_idx][1] - points[mcp_idx][1]) / hand_size
    
    return angle < angle_threshold and vertical_dist > distance_threshold

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
    'palm_distance': palm_distance_method,
    'xgboost': xgboost_method
}

def draw_finger_status(frame: np.ndarray, hand_data: Dict, method: Callable, landmarks: np.ndarray, 
                      position: tuple, color: tuple) -> None:
    """Draw finger detection results for a specific method."""
    h, w, _ = frame.shape
    
    def draw_point(point: np.ndarray, label: str = None, is_base: bool = False):
        """Helper to draw a point with optional label."""
        px = (point[:2] * [w, h]).astype(int)
        point_color = (0, 0, 255) if is_base else color  # Pure blue for base points
        cv2.circle(frame, px, 5, point_color, -1)  # Filled circle, slightly larger
        cv2.circle(frame, px, 5, (255, 255, 255), 1)  # White border
        if label == "Tip":  # Only show Tip labels
            cv2.putText(
                frame,
                label,
                (px[0] + 5, px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                2  # Thicker text
            )
        return px

    def draw_line(p1: np.ndarray, p2: np.ndarray):
        """Helper to draw a line between two points."""
        px1 = (p1[:2] * [w, h]).astype(int)
        px2 = (p2[:2] * [w, h]).astype(int)
        cv2.line(frame, px1, px2, color, 2)  # Thicker lines
    
    finger_indices = [
        ('Thumb', (4, 3, 2, 1)),
        ('Index', (8, 7, 6, 5)),
        ('Middle', (12, 11, 10, 9)),
        ('Ring', (16, 15, 14, 13)),
        ('Pinky', (20, 19, 18, 17))
    ]
    
    # Draw method-specific visualizations
    if method == palm_distance_method and _last_palm_center is not None:
        palm_center_px = draw_point(_last_palm_center)
        draw_line(landmarks[5], landmarks[17])
        draw_point(landmarks[5], is_base=True)
        draw_point(landmarks[17], is_base=True)
        
    elif method == adaptive_threshold_method:
        draw_line(landmarks[0], landmarks[9])
        palm_direction = landmarks[9] - landmarks[0]
        palm_end = landmarks[0] + palm_direction * 0.5
        draw_line(landmarks[0], palm_end)
        
    elif method == angle_based_method:
        pass
    
    # Calculate total extended fingers and build binary representation
    finger_states = []
    total_extended = 0
    
    for finger_name, (tip_idx, dip_idx, pip_idx, mcp_idx) in finger_indices:
        is_extended = method(landmarks, tip_idx, dip_idx, pip_idx, mcp_idx)
        if is_extended:
            total_extended += 1
        finger_states.append('1' if is_extended else '0')
        
        # Draw measurement points and lines specific to each method
        if method == palm_distance_method and _last_palm_center is not None:
            tip_px = draw_point(landmarks[tip_idx], "Tip")
            base_px = draw_point(landmarks[mcp_idx], is_base=True)
            palm_center_px = (_last_palm_center[:2] * [w, h]).astype(int)
            cv2.line(frame, palm_center_px, tip_px, color, 1)
            cv2.line(frame, palm_center_px, base_px, color, 1)
            
        elif method == adaptive_threshold_method:
            tip_px = draw_point(landmarks[tip_idx], "Tip")
            base_px = draw_point(landmarks[mcp_idx], is_base=True)
            finger_vector = landmarks[tip_idx] - landmarks[mcp_idx]
            finger_end = landmarks[mcp_idx] + finger_vector * 0.5
            draw_line(landmarks[mcp_idx], finger_end)
            
        elif method == angle_based_method:
            tip_px = draw_point(landmarks[tip_idx], "Tip")
            dip_px = draw_point(landmarks[dip_idx])
            pip_px = draw_point(landmarks[pip_idx])
            mcp_px = draw_point(landmarks[mcp_idx], is_base=True)
            cv2.line(frame, tip_px, dip_px, color, 1)
            cv2.line(frame, dip_px, pip_px, color, 1)
            cv2.line(frame, pip_px, mcp_px, color, 1)
    
    # Draw hand info and finger states
    handedness = hand_data.get('handedness', 'Unknown')
    cv2.putText(
        frame,
        f"{handedness} Hand - Count: {total_extended}",
        (position[0], position[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )
    
    # Draw binary representation
    cv2.putText(
        frame,
        f"States: {' '.join(finger_states)}",
        (position[0], position[1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        1
    )

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Initialize detector and analyzer
    detector = HandDetector()
    
    # Window setup for fullscreen
    window_name = 'Finger Detection Methods Comparison'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Get screen resolution using xrandr
    import subprocess
    try:
        output = subprocess.check_output('xrandr | grep "\*" | cut -d" " -f4', shell=True).decode()
        screen_width, screen_height = map(int, output.split('x'))
    except:
        screen_width, screen_height = 1920, 1080  # Default fallback
    
    cv2.resizeWindow(window_name, screen_width, screen_height)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
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
        
        # Resize frame to fill the screen
        frame = cv2.resize(frame, (screen_width, screen_height))

        # Add a semi-transparent overlay at the top for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (screen_width, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Detect hands
        frame, hands_data = detector.find_hands(frame, draw=True)

        # For each detected hand, show results from all methods
        for hand_idx, hand_data in enumerate(hands_data):
            landmarks = hand_data['landmarks']
            handedness = hand_data.get('handedness', 'Unknown')
            
            # Adjust x position based on handedness and screen width
            base_positions = {
                'Left': [int(screen_width * 0.02), int(screen_width * 0.15), int(screen_width * 0.28)],
                'Right': [int(screen_width * 0.55), int(screen_width * 0.7), int(screen_width * 0.85)]
            }
            x_positions = base_positions.get(handedness, [int(screen_width * 0.02), int(screen_width * 0.15), int(screen_width * 0.28)])
            
            # Different colors for each method - more vibrant colors
            colors = [
                (0, 255, 0),    # Pure green
                (255, 50, 50),  # Bright red
                (50, 50, 255)   # Bright blue
            ]
            
            for (method_name, method), color, x_pos in zip(DETECTION_METHODS.items(), colors, x_positions):
                # Draw method name with background
                text_size = cv2.getTextSize(method_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(
                    frame,
                    method_name,
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