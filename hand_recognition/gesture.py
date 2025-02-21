import numpy as np
from typing import Dict, List

class GestureAnalyzer:
    """Analyzes hand gestures using MediaPipe hand landmarks.
    
    The hand landmark model detects 21 3D landmarks on a hand:
    - Wrist: 0
    - Thumb: 1-4 (from base to tip)
    - Index: 5-8 (from base to tip)
    - Middle: 9-12 (from base to tip)
    - Ring: 13-16 (from base to tip)
    - Pinky: 17-20 (from base to tip)
    
    Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    """

    def __init__(self):
        """Initialize the GestureAnalyzer."""
        pass

    def get_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Angle in degrees
        """
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0)))

    def is_finger_extended(
        self,
        points: np.ndarray,
        tip_idx: int,
        dip_idx: int,
        pip_idx: int,
        mcp_idx: int
    ) -> bool:
        """Check if a finger is extended based on joint angles.
        
        Each finger (except thumb) has 4 landmarks:
        - MCP (Metacarpophalangeal) - Base joint
        - PIP (Proximal Interphalangeal) - First joint
        - DIP (Distal Interphalangeal) - Second joint
        - TIP - Fingertip
        
        Args:
            points: Array of hand landmarks
            tip_idx: Index of fingertip
            dip_idx: Index of DIP joint
            pip_idx: Index of PIP joint
            mcp_idx: Index of MCP joint
            
        Returns:
            True if finger is extended, False otherwise
        """
        # Get the three sections of the finger
        tip_to_dip = points[tip_idx] - points[dip_idx]
        dip_to_pip = points[dip_idx] - points[pip_idx]
        pip_to_mcp = points[pip_idx] - points[mcp_idx]
        
        # Calculate angles between sections
        angle1 = self.get_angle_between_vectors(tip_to_dip, dip_to_pip)
        angle2 = self.get_angle_between_vectors(dip_to_pip, pip_to_mcp)
        
        # A finger is extended if it's relatively straight
        return angle1 < 35 and angle2 < 35

    def is_thumb_extended(self, points: np.ndarray) -> bool:
        """Check if thumb is extended by comparing to index finger.
        
        Thumb has a different structure than other fingers:
        - CMC (Carpometacarpal) - 1
        - MCP (Metacarpophalangeal) - 2
        - IP (Interphalangeal) - 3
        - TIP - 4
        
        Args:
            points: Array of hand landmarks
            
        Returns:
            True if thumb is extended, False otherwise
        """
        # Get relevant points
        thumb_tip = points[4]    # Thumb tip
        thumb_base = points[2]   # Thumb MCP
        index_base = points[5]   # Index finger MCP
        
        # Calculate vectors
        thumb_vector = thumb_tip - thumb_base
        reference_vector = index_base - thumb_base
        
        # Check angle between thumb and reference vector
        angle = self.get_angle_between_vectors(thumb_vector, reference_vector)
        return angle > 45  # Thumb is extended if angle is large enough

    def count_fingers(self, landmarks: np.ndarray) -> int:
        """Count number of extended fingers.
        
        Args:
            landmarks: Array of hand landmarks from MediaPipe
            
        Returns:
            Number of extended fingers (0-5)
        """
        # Finger indices (tip, dip, pip, mcp)
        finger_indices = [
            (8, 7, 6, 5),    # Index finger landmarks
            (12, 11, 10, 9), # Middle finger landmarks
            (16, 15, 14, 13),# Ring finger landmarks
            (20, 19, 18, 17) # Pinky finger landmarks
        ]
        
        # Count extended fingers
        extended_count = 0
        
        # Check thumb separately due to different structure
        if self.is_thumb_extended(landmarks):
            extended_count += 1
            
        # Check other fingers
        for tip, dip, pip, mcp in finger_indices:
            if self.is_finger_extended(landmarks, tip, dip, pip, mcp):
                extended_count += 1
                
        return extended_count

    def analyze_hand(self, hand_data: Dict) -> Dict:
        """Analyze hand data and return gesture information.
        
        Args:
            hand_data: Dictionary containing hand landmarks and handedness
            
        Returns:
            Dictionary with analysis results including:
            - handedness: 'Left' or 'Right'
            - finger_count: Number of extended fingers (0-5)
            - confidence: Detection confidence
            - role: 'player' for left hand, 'points' for right hand
        """
        landmarks = hand_data['landmarks']
        handedness = hand_data['handedness']
        finger_count = self.count_fingers(landmarks)
        
        # Determine role based on handedness (left hand is player, right hand is points)
        role = "player" if handedness == "Left" else "points"
        
        return {
            'handedness': handedness,
            'finger_count': finger_count,
            'confidence': hand_data['confidence'],
            'role': role
        } 