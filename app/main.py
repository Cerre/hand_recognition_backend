from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import base64
import json
import logging
from typing import Dict, List, Any
import os
from collections import deque
from time import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI and middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://hand-tracker-web.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

class GestureStateTracker:
    def __init__(self, buffer_size=5, stable_threshold=0.6):
        self.buffer_size = buffer_size
        self.stable_threshold = stable_threshold
        # Store tuples of (timestamp, value)
        self.left_hand_buffer = deque(maxlen=buffer_size)
        self.right_hand_buffer = deque(maxlen=buffer_size)
        self.last_update_time = 0
        self.last_stable_state = {"player": None, "points": None}
        self.UPDATE_INTERVAL = 0.2  # 200ms between updates
        self.MAX_GESTURE_AGE = 2.0  # Maximum age of gestures to consider (2 seconds)

    def _clean_old_data(self, buffer, current_time):
        """Remove data older than MAX_GESTURE_AGE seconds"""
        cutoff_time = current_time - self.MAX_GESTURE_AGE
        return deque((ts, val) for ts, val in buffer if ts > cutoff_time)

    def add_frame_data(self, hands_data: List[Dict]) -> Dict:
        current_time = time()
        
        # Extract left and right hand data
        left_hand = None
        right_hand = None
        for hand in hands_data:
            if hand['handedness'] == 'Left':
                left_hand = hand['finger_count']
            else:
                right_hand = hand['finger_count']
        
        # Add to buffers with timestamps
        self.left_hand_buffer.append((current_time, left_hand))
        self.right_hand_buffer.append((current_time, right_hand))
        
        # Clean old data
        self.left_hand_buffer = self._clean_old_data(self.left_hand_buffer, current_time)
        self.right_hand_buffer = self._clean_old_data(self.right_hand_buffer, current_time)
        
        # Only process if enough time has passed
        if current_time - self.last_update_time < self.UPDATE_INTERVAL:
            return None
        
        # Get stable counts from recent data only
        player_num = self._get_stable_count(self.left_hand_buffer)
        points = self._get_stable_count(self.right_hand_buffer)
        
        # Send update if we have valid numbers
        if player_num is not None or points is not None:
            self.last_stable_state = {
                "player": player_num,
                "points": points
            }
            self.last_update_time = current_time
            
            return {
                "type": "score_update",
                "player": player_num,
                "points": points,
                "timestamp": current_time,
                "debug": {
                    "left_buffer_size": len(self.left_hand_buffer),
                    "right_buffer_size": len(self.right_hand_buffer),
                    "data_age": current_time - min(
                        (ts for ts, _ in self.left_hand_buffer), 
                        default=current_time
                    )
                }
            }
        
        return None

    def _get_stable_count(self, buffer) -> int:
            """Return most frequent number in buffer"""
            if not buffer or len(buffer) < 2:
                return None
                
            # Extract only values from timestamp-value pairs
            valid_counts = [val for _, val in buffer if val is not None]
            if len(valid_counts) < 2:
                return None
                
            # Check last two readings for quick response
            if valid_counts[-1] == valid_counts[-2]:
                return int(valid_counts[-1])
                
            # Fallback to frequency-based detection
            count_freq = {}
            for count in valid_counts:
                count_freq[count] = count_freq.get(count, 0) + 1
            
            max_count = max(count_freq.items(), key=lambda x: x[1])
            
            if max_count[1] / len(valid_counts) >= self.stable_threshold:
                return int(max_count[0])
                
            return None

def count_fingers(landmarks) -> dict:
    """
    Simple finger counter - just checks if fingers are raised
    """
    # Landmark indices for fingertips and base joints
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    finger_bases = [2, 5, 9, 13, 17]  # Base joints
    
    # Convert landmarks to numpy array
    points = np.array([[l.x, l.y, l.z] for l in landmarks])
    
    # Count extended fingers - a finger is extended if its tip is higher than its base
    extended_fingers = sum(
        1 for tip_idx, base_idx in zip(finger_tips, finger_bases)
        if points[tip_idx][1] < points[base_idx][1]  # Y coordinate is less means higher up
    )
    
    # Adjust count to fix the off-by-one error (subtract 1 if we detected any fingers)
    final_count = max(0, extended_fingers - 1) if extended_fingers > 0 else 0
    
    return {
        "total_count": int(final_count)
    }

def process_frame(base64_frame: str) -> Dict[str, Any]:
    """Process a single frame and detect hands"""
    try:
        # Decode base64 image
        img_data = base64.b64decode(base64_frame.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"hands": []}

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Process detected hands
        hand_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get finger data
                finger_data = count_fingers(hand_landmarks.landmark)
                
                hand_data.append({
                    'handedness': str(handedness.classification[0].label),
                    'finger_count': int(finger_data['total_count'])
                })
                
        return {"hands": hand_data}
        
    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}", exc_info=True)
        return {"hands": []}

@app.get("/")
async def root():
    logger.info("Health check endpoint accessed")
    return {"message": "Hand recognition backend is running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)
    logger.info(f"New client {client_id} attempting to connect")
    
    # Initialize state tracker for this connection
    state_tracker = GestureStateTracker()
    
    try:
        await websocket.accept()
        logger.info(f"Client {client_id} connected successfully")
        
        while True:
            try:
                data = await websocket.receive_text()
                
                if ',' not in data:
                    await websocket.send_json({
                        "error": "Invalid data format",
                        "hands": []
                    })
                    continue
                
                # Process frame and update state
                frame_data = process_frame(data)
                update = state_tracker.add_frame_data(frame_data["hands"])
                
                # Only send update if there's a stable change
                if update:
                    await websocket.send_json(update)
                
            except WebSocketDisconnect:
                logger.warning(f"Client {client_id} disconnected")
                break
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON encoding error for client {client_id}: {str(e)}")
                await websocket.send_json({
                    "error": "Failed to encode response",
                    "hands": []
                })
                
            except Exception as e:
                logger.error(f"Error processing frame for client {client_id}: {str(e)}", exc_info=True)
                try:
                    await websocket.send_json({
                        "error": "Internal server error",
                        "hands": []
                    })
                except:
                    logger.error(f"Failed to send error message to client {client_id}")
                    break
                    
    except Exception as e:
        logger.error(f"Failed to establish WebSocket connection for client {client_id}: {str(e)}", exc_info=True)
        
    finally:
        logger.info(f"WebSocket connection closed for client {client_id}")