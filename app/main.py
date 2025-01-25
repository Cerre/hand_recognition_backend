from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import base64
import json
import logging
from typing import Dict, List, Any
import hashlib
import hmac
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
# Get API key from environment variable
API_KEY = os.getenv("API_KEY", "124")  # Default only for development
SALT = os.getenv("SALT", os.urandom(16).hex())  # Generate random salt if not provided

# Pre-compute the hash of the API key
def hash_api_key(key: str, salt: str) -> str:
    """Hash the API key using SHA-256 and a salt"""
    return hashlib.pbkdf2_hmac(
        'sha256', 
        key.encode(), 
        salt.encode(), 
        100000  # Number of iterations
    ).hex()

STORED_API_KEY_HASH = hash_api_key(API_KEY, SALT)

# Define allowed origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://hand-recognition-frontend.vercel.app/",
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

def verify_api_key(provided_key: str) -> bool:
    """Verify a provided API key against the stored hash"""
    if not provided_key:
        return False
    
    # Hash the provided key with the same salt
    provided_key_hash = hash_api_key(provided_key, SALT)
    
    # Compare in constant time to prevent timing attacks
    return hmac.compare_digest(
        provided_key_hash.encode(),
        STORED_API_KEY_HASH.encode()
    )

async def authenticate_websocket(websocket: WebSocket) -> bool:
    """Authenticate WebSocket connection using API key"""
    api_key = websocket.query_params.get("api_key")
    
    if not verify_api_key(api_key):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logger.warning("Invalid API key attempt")
        return False
        
    return True

def process_frame(base64_frame: str) -> Dict[str, List[Dict[str, Any]]]:
    """Process a single frame and detect hands."""
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
            
            for hand_landmarks in results.multi_hand_landmarks:
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
                
                hand_data.append({
                    'landmarks': landmarks,
                    'bbox': bbox
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
        # Authenticate before accepting connection
        if not await authenticate_websocket(websocket):
            return
            
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
                    logger.debug(f"Sent hand data to client {client_id}: {len(response_data['hands'])} hands detected")
                
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