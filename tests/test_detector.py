import numpy as np
import cv2
import pytest
from hand_recognition.detector import HandDetector

class TestHandDetector:
    @pytest.fixture
    def detector(self):
        return HandDetector()

    def test_detector_initialization(self, detector):
        """Test that the detector initializes with correct components"""
        assert hasattr(detector, 'hands'), "Detector should have hands attribute"
        assert hasattr(detector, 'mp_draw'), "Detector should have mp_draw attribute"
        assert detector.mp_hands is not None, "MediaPipe hands should be initialized"

    def test_landmark_coordinate_conversion(self, detector):
        """Test conversion from normalized coordinates to pixel coordinates"""
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create test landmarks (normalized coordinates)
        landmarks = np.array([
            [0.5, 0.5, 0],  # Center
            [0.0, 0.0, 0],  # Top-left
            [1.0, 1.0, 0],  # Bottom-right
        ])
        
        # Convert to pixel coordinates
        pixel_coords = detector.get_landmark_coords(frame, landmarks)
        
        # Check center point
        assert pixel_coords[0][0] == 320, "X coordinate should be half of frame width"
        assert pixel_coords[0][1] == 240, "Y coordinate should be half of frame height"
        
        # Check corners
        assert np.all(pixel_coords[1] == [0, 0]), "Top-left should be at (0,0)"
        assert np.all(pixel_coords[2] == [640, 480]), "Bottom-right should be at (width,height)"

    def test_find_hands_empty_frame(self, detector):
        """Test hand detection on an empty frame"""
        # Create an empty frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame
        processed_frame, hands_data = detector.find_hands(frame)
        
        # Check that the function returns expected types
        assert isinstance(processed_frame, np.ndarray), "Should return numpy array for frame"
        assert isinstance(hands_data, list), "Should return list for hands data"
        assert len(hands_data) == 0, "Should detect no hands in empty frame"
        
        # Check frame dimensions haven't changed
        assert processed_frame.shape == (480, 640, 3), "Frame dimensions should remain unchanged"

    def test_find_hands_output_format(self, detector):
        """Test that find_hands returns data in the format expected by the backend"""
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame
        _, hands_data = detector.find_hands(frame)
        
        # Even with no hands, verify the output format
        assert isinstance(hands_data, list), "Output should be a list"
        
        # If any hands were detected (not guaranteed), verify their format
        for hand in hands_data:
            assert isinstance(hand, dict), "Each hand should be a dictionary"
            assert 'landmarks' in hand, "Hand should contain landmarks"
            assert 'handedness' in hand, "Hand should contain handedness"
            assert 'confidence' in hand, "Hand should contain confidence"
            
            assert isinstance(hand['landmarks'], np.ndarray), "Landmarks should be numpy array"
            assert hand['landmarks'].shape[1] == 3, "Landmarks should have 3 coordinates (x,y,z)"
            assert isinstance(hand['handedness'], str), "Handedness should be string"
            assert isinstance(hand['confidence'], float), "Confidence should be float" 