from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import logging
from typing import Dict, List, Any
import os
from time import time

# Import hand_recognition package
from hand_recognition import HandDetector
from tools.xgboost_predictor import xgboost_method

# Set up logging - Change to WARNING to reduce overhead
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING
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

# Initialize hand detection components
detector = HandDetector(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def count_fingers(landmarks: np.ndarray) -> int:
    """Count number of extended fingers using XGBoost method."""
    finger_indices = [
        (4, 3, 2, 1),    # Thumb
        (8, 7, 6, 5),    # Index
        (12, 11, 10, 9), # Middle
        (16, 15, 14, 13),# Ring
        (20, 19, 18, 17) # Pinky
    ]
    
    extended = 0
    for tip_idx, dip_idx, pip_idx, mcp_idx in finger_indices:
        if xgboost_method(landmarks, tip_idx, dip_idx, pip_idx, mcp_idx):
            extended += 1
    
    return extended

class GestureStateTracker:
    def __init__(self):
        pass  # No initialization needed

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
        
        logger.debug(f"Processed frame data - Left hand: {left_hand}, Right hand: {right_hand}")
        
        # Return immediate update
        return {
            "type": "score_update",
            "player": right_hand,   # Player number from right hand
            "points": left_hand,    # Points from left hand
            "timestamp": time()
        }

def process_frame(base64_frame: str) -> Dict[str, Any]:
    """Process a single frame and detect hands"""
    try:
        # Quick validation of base64 data
        if not base64_frame or ',' not in base64_frame:
            return {"hands": []}

        # Decode base64 image
        try:
            img_data = base64.b64decode(base64_frame.split(',')[1])
            if not img_data:
                return {"hands": []}
        except Exception as e:
            logger.warning(f"Base64 decoding failed: {str(e)}")
            return {"hands": []}

        # Convert to numpy array and decode image
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"hands": []}

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use HandDetector to find hands
        frame, hands_data = detector.find_hands(frame_rgb)
        
        # Process each hand with XGBoost method
        processed_hands = []
        for hand_data in hands_data:
            finger_count = count_fingers(hand_data['landmarks'])
            processed_hands.append({
                'handedness': hand_data['handedness'],
                'finger_count': finger_count
            })

        return {"hands": processed_hands}

    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        return {"hands": []}

@app.get("/")
async def root():
    logger.info("Health check endpoint accessed")
    return {"message": "Hand recognition backend is running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)
    logger.info(f"New client {client_id} connected from {websocket.client.host}")
    
    state_tracker = GestureStateTracker()
    frame_count = 0
    last_frame_time = time()
    
    try:
        await websocket.accept()
        await websocket.send_json({"status": "connected"})
        
        while True:
            try:
                # Process frame
                data = await websocket.receive_text()
                current_time = time()
                
                # Only log if there's a significant delay
                if current_time - last_frame_time > 5:
                    logger.warning(f"Long delay between frames: {current_time - last_frame_time:.2f}s")
                
                last_frame_time = current_time
                frame_count += 1
                
                # Process frame and send update
                frame_data = process_frame(data)
                update = state_tracker.add_frame_data(frame_data["hands"])
                await websocket.send_json(update)
                
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected after {frame_count} frames")
                break
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                try:
                    await websocket.send_json({"error": "Internal server error", "hands": []})
                except:
                    break
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info(f"Connection closed. Frames processed: {frame_count}")