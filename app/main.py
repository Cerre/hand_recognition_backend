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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Key configuration
API_KEY = os.environ.get("API_KEY", "default-key-for-development")

# Define allowed origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://hand-tracker-web.vercel.app",
]

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
    min_detection_confidence=0.5
)

def count_fingers(landmarks) -> dict:
    """
    Count extended fingers using hand landmarks
    Returns dictionary with finger states and total count
    """
    # Finger landmark indices
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_bases = [2, 5, 9, 13, 17]  # For non-thumb fingers
    
    # Track each finger's state
    finger_states = {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False
    }
    
    # Convert landmarks to numpy array for easier calculations
    points = np.array([[l.x, l.y, l.z] for l in landmarks])
    
    # Check each finger
    # Special case for thumb
    thumb_tip = points[finger_tips[0]]
    thumb_base = points[finger_bases[0]]
    thumb_ip = points[3]  # Inner thumb joint
    
    # Check if thumb is extended based on the angle between tip, IP, and base
    v1 = thumb_tip - thumb_ip
    v2 = thumb_base - thumb_ip
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    finger_states["thumb"] = angle > 1.0  # ~57 degrees
    
    # For other fingers, check if tip is higher than base (in y-coordinate)
    finger_names = ["index", "middle", "ring", "pinky"]
    for idx, (name, tip_idx, base_idx) in enumerate(zip(
        finger_names,
        finger_tips[1:],
        finger_bases[1:]
    )):
        # Finger is extended if tip is higher than base (lower y value)
        finger_states[name] = points[tip_idx][1] < points[base_idx][1]
    
    # Count total extended fingers
    total_count = sum(1 for state in finger_states.values() if state)
    
    return {
        "finger_states": finger_states,
        "total_count": total_count
    }

def process_frame(base64_frame: str) -> Dict[str, List[Dict[str, Any]]]:
    """Process a single frame and detect hands and count fingers."""
    try:
        # Decode base64 image
        img_data = base64.b64decode(base64_frame.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image frame")
            return {"hands": []}

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Process detected hands
        hand_data = []
        if results.multi_hand_landmarks:
            image_height, image_width, _ = frame.shape
            logger.info(f"Detected {len(results.multi_hand_landmarks)} hands")
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks = []
                x_coords = []
                y_coords = []
                
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': float(landmark.x),
                        'y': float(landmark.y),
                        'z': float(landmark.z)
                    })
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
                
                bbox = {
                    'x_min': min(x_coords),
                    'x_max': max(x_coords),
                    'y_min': min(y_coords),
                    'y_max': max(y_coords)
                }
                
                # Count fingers for this hand
                finger_data = count_fingers(hand_landmarks.landmark)
                
                hand_data.append({
                    'landmarks': landmarks,
                    'bbox': bbox,
                    'handedness': handedness.classification[0].label,  # 'Left' or 'Right'
                    'finger_count': finger_data['total_count'],
                    'finger_states': finger_data['finger_states']
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
    
    try:
        await websocket.accept()
        logger.info(f"Client {client_id} connected successfully")
        
        while True:
            try:
                data = await websocket.receive_text()
                logger.debug(f"Received frame from client {client_id}")
                
                if ',' not in data:
                    logger.error(f"Client {client_id} sent invalid data format")
                    await websocket.send_json({
                        "error": "Invalid data format",
                        "hands": []
                    })
                    continue
                
                response_data = process_frame(data)
                await websocket.send_json(response_data)
                
                if response_data["hands"]:
                    logger.debug(f"Processed hand data: {response_data}")
                
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