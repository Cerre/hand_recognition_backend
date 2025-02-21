import numpy as np
import pytest
from hand_recognition.gesture import GestureAnalyzer

class TestGestureAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return GestureAnalyzer()

    def test_hand_role_assignment(self, analyzer):
        """Test that hand roles are correctly assigned (left=player, right=points)"""
        # Test left hand
        left_hand_data = {
            'landmarks': np.zeros((21, 3)),  # Dummy landmarks
            'handedness': 'Left',
            'confidence': 0.9
        }
        left_result = analyzer.analyze_hand(left_hand_data)
        assert left_result['role'] == 'player', "Left hand should be assigned player role"

        # Test right hand
        right_hand_data = {
            'landmarks': np.zeros((21, 3)),  # Dummy landmarks
            'handedness': 'Right',
            'confidence': 0.9
        }
        right_result = analyzer.analyze_hand(right_hand_data)
        assert right_result['role'] == 'points', "Right hand should be assigned points role"

    def test_finger_counting_all_extended(self, analyzer):
        """Test counting fingers when all are extended"""
        # Create landmarks for a hand with all fingers extended
        landmarks = np.zeros((21, 3))
        
        # Set wrist position
        landmarks[0] = [0.5, 0.5, 0]
        
        # Thumb landmarks (CMC, MCP, IP, TIP)
        thumb = np.array([
            [0.4, 0.5, 0],    # CMC
            [0.3, 0.4, 0],    # MCP
            [0.2, 0.3, 0],    # IP
            [0.1, 0.2, 0],    # TIP
        ])
        landmarks[1:5] = thumb
        
        # Other fingers - straight up position
        for finger in range(4):  # Index, Middle, Ring, Pinky
            base_idx = 5 + (finger * 4)
            # Create straight line for each finger
            finger_landmarks = np.array([
                [0.5, 0.5, 0],  # MCP - at wrist level
                [0.5, 0.6, 0],  # PIP - above
                [0.5, 0.7, 0],  # DIP - above
                [0.5, 0.8, 0],  # TIP - top
            ])
            landmarks[base_idx:base_idx + 4] = finger_landmarks
        
        hand_data = {
            'landmarks': landmarks,
            'handedness': 'Left',
            'confidence': 0.9
        }
        
        result = analyzer.analyze_hand(hand_data)
        assert result['finger_count'] == 5, "Should detect all 5 fingers when extended"

    def test_finger_counting_all_closed(self, analyzer):
        """Test counting fingers when all are closed (fist)"""
        # Create landmarks for a closed fist
        landmarks = np.zeros((21, 3))
        
        # Set all finger tips below other joints (closed fist position)
        for tip_idx in [4, 8, 12, 16, 20]:  # All tips
            landmarks[tip_idx, 1] = -0.5  # Tips below
            
        # Set middle joints
        for mid_idx in [3, 7, 11, 15, 19]:
            landmarks[mid_idx, 1] = 0.0
            
        hand_data = {
            'landmarks': landmarks,
            'handedness': 'Right',
            'confidence': 0.9
        }
        
        result = analyzer.analyze_hand(hand_data)
        assert result['finger_count'] == 0, "Should detect 0 fingers when all are closed"

    def test_backend_format_compatibility(self, analyzer):
        """Test that the analyzer output matches the backend's expected format"""
        hand_data = {
            'landmarks': np.zeros((21, 3)),
            'handedness': 'Left',
            'confidence': 0.9
        }
        
        result = analyzer.analyze_hand(hand_data)
        
        # Check all required fields are present
        assert 'handedness' in result, "Result should contain handedness"
        assert 'finger_count' in result, "Result should contain finger_count"
        assert 'confidence' in result, "Result should contain confidence"
        assert 'role' in result, "Result should contain role"
        
        # Check data types
        assert isinstance(result['finger_count'], int), "finger_count should be an integer"
        assert 0 <= result['finger_count'] <= 5, "finger_count should be between 0 and 5"
        assert isinstance(result['handedness'], str), "handedness should be a string"
        assert isinstance(result['role'], str), "role should be a string"
        assert isinstance(result['confidence'], float), "confidence should be a float"

    def test_angle_calculation(self, analyzer):
        """Test the angle calculation between vectors"""
        v1 = np.array([1, 0, 0])  # Vector along x-axis
        v2 = np.array([0, 1, 0])  # Vector along y-axis
        
        angle = analyzer.get_angle_between_vectors(v1, v2)
        assert abs(angle - 90.0) < 0.001, "Perpendicular vectors should have 90 degree angle"
        
        # Test parallel vectors
        v3 = np.array([1, 0, 0])
        angle = analyzer.get_angle_between_vectors(v1, v3)
        assert abs(angle - 0.0) < 0.001, "Parallel vectors should have 0 degree angle" 