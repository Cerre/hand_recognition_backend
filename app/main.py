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
    def __init__(self):
        pass  # No initialization needed anymore

    def add_frame_data(self, hands_data: List[Dict]) -> Dict:
        """Simply process the current frame data and return results"""
        # Extract left and right hand data
        left_hand = None
        right_hand = None
        
        for hand in hands_data:
            if hand['handedness'] == 'Left':
                left_hand = hand['finger_count']
            else:
                right_hand = hand['finger_count']
        
        # Return immediate update
        return {
            "type": "score_update",
            "player": left_hand,   # Player number from left hand
            "points": right_hand,  # Points from right hand
            "timestamp": time()
        }

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
                
                # Process frame and send immediate update
                frame_data = process_frame(data)
                update = state_tracker.add_frame_data(frame_data["hands"])
                await websocket.send_json(update)
                
            except WebSocketDisconnect:
                logger.warning(f"Client {client_id} disconnected")
                break
                
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