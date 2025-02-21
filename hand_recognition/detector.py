import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

class HandDetector:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7
    ):
        """Initialize the HandDetector with MediaPipe Hands.
        
        Args:
            static_image_mode: Whether to treat input images as a video stream or independent images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def find_hands(self, frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Detect hands in the frame and optionally draw landmarks.
        
        Args:
            frame: Input image/video frame (BGR format)
            draw: Whether to draw the landmarks on the frame
            
        Returns:
            Tuple containing:
            - Processed frame with landmarks drawn (if draw=True)
            - List of dictionaries containing hand data
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(frame_rgb)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Draw landmarks if requested
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Convert landmarks to numpy array
                landmarks = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])
                
                # Store hand data
                hands_data.append({
                    'landmarks': landmarks,
                    'handedness': handedness.classification[0].label,
                    'confidence': handedness.classification[0].score
                })
        
        return frame, hands_data
    
    def get_landmark_coords(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates.
        
        Args:
            frame: Input frame
            landmarks: Normalized landmarks array
            
        Returns:
            Array of landmark coordinates in pixels
        """
        h, w, _ = frame.shape
        return np.multiply(landmarks[:, :2], [w, h]).astype(int) 