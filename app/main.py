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
    def __init__(self, buffer_size=10, stable_threshold=0.7):
        self.buffer_size = buffer_size
        self.stable_threshold = stable_threshold
        self.left_hand_buffer = deque(maxlen=buffer_size)
        self.right_hand_buffer = deque(maxlen=buffer_size)
        self.last_update_time = 0
        self.last_stable_state = {"player": None, "points": None}
        self.UPDATE_INTERVAL = 0.5  # seconds between updates

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
        
        # Add to buffers
        self.left_hand_buffer.append(left_hand)
        self.right_hand_buffer.append(right_hand)
        
        # Only process if enough time has passed
        if current_time - self.last_update_time < self.UPDATE_INTERVAL:
            return None
        
        # Get stable counts
        player_num = self._get_stable_count(self.left_hand_buffer)
        points = self._get_stable_count(self.right_hand_buffer)
        
        # Check if state has changed
        if (player_num != self.last_stable_state["player"] or 
            points != self.last_stable_state["points"]):
            
            self.last_stable_state = {
                "player": player_num,
                "points": points
            }
            self.last_update_time = current_time
            
            return {
                "type": "score_update",
                "player": player_num,
                "points": points,
                "timestamp": current_time
            }
        
        return None

    def _get_stable_count(self, buffer) -> int:
        """Return most frequent number in buffer if it meets stability threshold"""
        if not buffer:
            return None
            
        # Filter out None values
        valid_counts = [x for x in buffer if x is not None]
        if not valid_counts:
            return None
            
        # Count frequencies
        count_freq = {}
        for count in valid_counts:
            count_freq[count] = count_freq.get(count, 0) + 1
        
        # Find most frequent
        max_count = max(count_freq.items(), key=lambda x: x[1])
        
        # Check if meets stability threshold
        if max_count[1] / len(buffer) >= self.stable_threshold:
            return int(max_count[0])  # Convert to Python int
            
        return None

def count_fingers(landmarks) -> dict:
    """Count extended fingers using hand landmarks"""
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_bases = [2, 5, 9, 13, 17]  # For non-thumb fingers
    
    finger_states = {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False
    }
    
    # Convert landmarks to numpy array for easier calculations
    points = np.array([[l.x, l.y, l.z] for l in landmarks])
    
    # Special case for thumb
    thumb_tip = points[finger_tips[0]]
    thumb_ip = points[3]
    thumb_base = points[finger_bases[0]]
    
    v1 = thumb_tip - thumb_ip
    v2 = thumb_base - thumb_ip
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    finger_states["thumb"] = bool(angle > 1.0)  # Convert to Python bool
    
    # For other fingers
    finger_names = ["index", "middle", "ring", "pinky"]
    for name, tip_idx, base_idx in zip(finger_names, finger_tips[1:], finger_bases[1:]):
        finger_states[name] = bool(points[tip_idx][1] < points[base_idx][1])  # Convert to Python bool
    
    total_count = sum(1 for state in finger_states.values() if state)
    
    return {
        "finger_states": finger_states,
        "total_count": int(total_count)  # Convert to Python int
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
                    'handedness': str(handedness.classification[0].label),  # Convert to Python str
                    'finger_count': int(finger_data['total_count']),  # Convert to Python int
                    'finger_states': {k: bool(v) for k, v in finger_data['finger_states'].items()}  # Convert to Python bool
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