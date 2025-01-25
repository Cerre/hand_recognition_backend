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
    Count extended fingers including thumb using MediaPipe hand landmarks.
    Uses joint angles to detect extended fingers.
    """
    # Convert landmarks to numpy array
    points = np.array([[l.x, l.y, l.z] for l in landmarks])
    
    def get_angle_between_vectors(v1, v2):
        """Calculate angle between two vectors in degrees"""
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0)))

    def is_finger_extended(tip_idx, dip_idx, pip_idx, mcp_idx):
        """Check if a finger is extended based on joint angles"""
        # Get the three sections of the finger
        tip_to_dip = points[tip_idx] - points[dip_idx]
        dip_to_pip = points[dip_idx] - points[pip_idx]
        pip_to_mcp = points[pip_idx] - points[mcp_idx]
        
        # Calculate angles between sections
        angle1 = get_angle_between_vectors(tip_to_dip, dip_to_pip)
        angle2 = get_angle_between_vectors(dip_to_pip, pip_to_mcp)
        
        # A finger is extended if it's relatively straight
        return angle1 < 35 and angle2 < 35

    def is_thumb_extended(points):
        """Check if thumb is extended using its unique joint structure"""
        # Thumb landmark indices: 4 (tip), 3, 2, 1 (base)
        tip_to_ip = points[4] - points[3]  # IP: Interphalangeal joint
        ip_to_mcp = points[3] - points[2]  # MCP: Metacarpophalangeal joint
        mcp_to_cmc = points[2] - points[1]  # CMC: Carpometacarpal joint
        
        # Calculate angles
        angle1 = get_angle_between_vectors(tip_to_ip, ip_to_mcp)
        angle2 = get_angle_between_vectors(ip_to_mcp, mcp_to_cmc)
        
        # Thumb has different threshold due to its natural position
        return angle1 < 45 and angle2 < 45
    
    # Check thumb
    thumb_extended = is_thumb_extended(points)
    
    # Check each finger
    fingers_extended = [
        thumb_extended,  # Thumb
        is_finger_extended(8, 7, 6, 5),    # Index
        is_finger_extended(12, 11, 10, 9),  # Middle
        is_finger_extended(16, 15, 14, 13), # Ring
        is_finger_extended(20, 19, 18, 17)  # Pinky
    ]
    
    # Count total extended fingers
    count = sum(1 for extended in fingers_extended if extended)
    
    return {
        "total_count": int(count)
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