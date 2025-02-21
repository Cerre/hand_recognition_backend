import pytest
from fastapi.testclient import TestClient
import base64
import cv2
import numpy as np
from app.main import app, process_frame, GestureStateTracker

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def test_frame():
    """Create a test frame with a solid color"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (200, 200, 200)  # Light gray color
    _, buffer = cv2.imencode('.jpg', frame)
    base64_frame = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    return base64_frame

def test_health_check(test_client):
    """Test the health check endpoint"""
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hand recognition backend is running"}

def test_process_frame_empty():
    """Test processing an empty frame"""
    result = process_frame("data:,")
    assert result == {"hands": []}

def test_process_frame_invalid_base64():
    """Test processing an invalid base64 frame"""
    result = process_frame("invalid_data")
    assert result == {"hands": []}

def test_process_frame_valid(test_frame):
    """Test processing a valid frame"""
    result = process_frame(test_frame)
    assert "hands" in result
    assert isinstance(result["hands"], list)

def test_gesture_state_tracker():
    """Test the GestureStateTracker class"""
    tracker = GestureStateTracker()
    
    # Test with no hands
    update = tracker.add_frame_data([])
    assert update["type"] == "score_update"
    assert update["player"] is None
    assert update["points"] is None
    
    # Test with one hand (right)
    update = tracker.add_frame_data([
        {"handedness": "Right", "finger_count": 3}
    ])
    assert update["type"] == "score_update"
    assert update["player"] == 3
    assert update["points"] is None
    
    # Test with both hands
    update = tracker.add_frame_data([
        {"handedness": "Left", "finger_count": 4},
        {"handedness": "Right", "finger_count": 2}
    ])
    assert update["type"] == "score_update"
    assert update["player"] == 2
    assert update["points"] == 4

@pytest.mark.asyncio
async def test_websocket_connection(test_client):
    """Test WebSocket connection and initial message"""
    with test_client.websocket_connect("/ws") as websocket:
        data = websocket.receive_json()
        assert data == {"status": "connected"}

@pytest.mark.asyncio
async def test_websocket_frame_processing(test_client, test_frame):
    """Test WebSocket frame processing"""
    with test_client.websocket_connect("/ws") as websocket:
        # Skip initial connection message
        websocket.receive_json()
        
        # Send a test frame
        websocket.send_text(test_frame)
        
        # Get the response
        response = websocket.receive_json()
        assert "type" in response
        assert response["type"] == "score_update"
        assert "player" in response
        assert "points" in response
        assert "timestamp" in response 