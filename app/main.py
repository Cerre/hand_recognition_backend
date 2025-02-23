from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import cv2
import numpy as np
import base64
import logging
from typing import Dict, List, Any
import os
import json
from time import time
import hashlib
import hmac
from collections import defaultdict
import asyncio

# Import hand_recognition package
from hand_recognition import HandDetector
from tools.xgboost_predictor import xgboost_method

# Set up logging - Change to WARNING to reduce overhead
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get API key from environment
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    logger.warning("No API_KEY environment variable set")

# Initialize FastAPI and middlewares
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

# Security setup
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting setup
connection_attempts = defaultdict(list)  # IP -> list of timestamps
MAX_ATTEMPTS = 5  # Maximum connection attempts
RATE_LIMIT_WINDOW = 60  # Time window in seconds
RATE_LIMIT_CLEANUP_INTERVAL = 300  # Cleanup old entries every 5 minutes
last_cleanup = time()

def create_connection_token(timestamp: str) -> str:
    """Create a secure connection token using HMAC"""
    if not API_KEY:
        return ""
    return hmac.new(
        API_KEY.encode(),
        timestamp.encode(),
        hashlib.sha256
    ).hexdigest()

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
        self.last_update_time = time()
        self.min_update_interval = 1.0 / 30  # Cap at 30 FPS to reduce load

    def add_frame_data(self, hands_data: List[Dict]) -> Dict:
        """Process the current frame data and return results"""
        current_time = time()
        time_since_last_update = current_time - self.last_update_time
        
        # If we're getting updates too quickly, return the last state
        if time_since_last_update < self.min_update_interval:
            return None
            
        # Extract left and right hand data
        left_hand = None
        right_hand = None
        
        for hand in hands_data:
            if hand['handedness'] == 'Left':
                left_hand = hand['finger_count']
            else:
                right_hand = hand['finger_count']
        
        self.last_update_time = current_time
        
        # Return immediate update
        return {
            "type": "score_update",
            "player": right_hand,   # Player number from right hand
            "points": left_hand,    # Points from left hand
            "timestamp": current_time
        }

def process_frame(base64_frame: str) -> Dict[str, Any]:
    """Process a single frame and detect hands"""
    try:
        # Handle empty frames during initialization more gracefully
        if not base64_frame:
            return {"hands": [], "status": "waiting_for_camera"}
        if base64_frame == "data:,":
            return {"hands": [], "status": "camera_initializing"}

        # Quick validation of base64 data
        if ',' not in base64_frame:
            return {"hands": [], "status": "invalid_frame"}

        # Decode base64 image
        try:
            img_data = base64.b64decode(base64_frame.split(',')[1])
            if not img_data:
                return {"hands": [], "status": "empty_frame"}
        except Exception as e:
            logger.warning(f"Base64 decoding failed: {str(e)}")
            return {"hands": [], "status": "decode_error"}

        # Convert to numpy array and decode image
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"hands": [], "status": "invalid_image"}

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

        return {"hands": processed_hands, "status": "ok"}

    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        return {"hands": [], "status": "error"}

@app.get("/")
async def root():
    logger.info("Health check endpoint accessed")
    return {"message": "Hand recognition backend is running"}

def is_rate_limited(ip: str, current_time: float) -> bool:
    """Check if an IP has exceeded rate limits"""
    global last_cleanup
    
    # Cleanup old entries periodically
    if current_time - last_cleanup > RATE_LIMIT_CLEANUP_INTERVAL:
        for ip_addr in list(connection_attempts.keys()):
            # Remove attempts older than the window
            connection_attempts[ip_addr] = [
                t for t in connection_attempts[ip_addr]
                if current_time - t < RATE_LIMIT_WINDOW
            ]
            # Remove empty entries
            if not connection_attempts[ip_addr]:
                del connection_attempts[ip_addr]
        last_cleanup = current_time
    
    # Get attempts within the time window
    recent_attempts = [
        t for t in connection_attempts[ip]
        if current_time - t < RATE_LIMIT_WINDOW
    ]
    
    # Update attempts list
    connection_attempts[ip] = recent_attempts
    connection_attempts[ip].append(current_time)
    
    # Check if too many attempts
    return len(recent_attempts) >= MAX_ATTEMPTS

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)
    client_ip = websocket.client.host
    current_time = time()
    
    # Check rate limit before accepting connection
    if is_rate_limited(client_ip, current_time):
        logger.warning(f"Rate limit exceeded for IP {client_ip}")
        return
        
    logger.info(f"New client {client_id} attempting to connect from {client_ip}")
    
    try:
        # Accept the connection first
        await websocket.accept()
        logger.info(f"Connection accepted for client {client_id}, waiting for authentication")
        
        # Set a timeout for authentication
        auth_timeout = 5  # seconds
        auth_start = time()
        
        # Wait for authentication message
        try:
            while True:
                if time() - auth_start > auth_timeout:
                    logger.warning(f"Authentication timeout for client {client_id}")
                    await websocket.close(1008, "Authentication timeout")
                    return
                    
                try:
                    # Try to receive the authentication message
                    auth_message = await websocket.receive_json()
                    break
                except Exception:
                    # If we get any error, wait a bit and try again if within timeout
                    await asyncio.sleep(0.1)
                    continue
            
            # Validate message format
            if not isinstance(auth_message, dict) or 'key' not in auth_message:
                logger.warning(f"Client {client_id} sent invalid auth format")
                await websocket.close(1008, "Invalid authentication format")
                return
                
            # Validate API key
            client_key = auth_message['key']
            if not API_KEY:
                logger.error("Server API_KEY not configured")
                await websocket.close(1011, "Server configuration error")
                return
                
            if client_key != API_KEY:
                logger.warning(f"Client {client_id} failed authentication")
                await websocket.close(1008, "Invalid authentication")
                return
                
            logger.info(f"Client {client_id} authenticated successfully")
            
            # Send success message
            await websocket.send_json({
                "status": "connected",
                "message": "Authentication successful"
            })
            
        except Exception as e:
            logger.warning(f"Client {client_id} authentication error: {str(e)}")
            await websocket.close(1008, "Authentication error")
            return
    
        # Initialize video processing
        state_tracker = GestureStateTracker()
        frame_count = 0
        last_frame_time = time()
        connection_start_time = time()
        
        # Main processing loop
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
                
                # Process frame
                frame_data = process_frame(data)
                
                # During first few seconds, send more detailed status updates
                if current_time - connection_start_time < 5:
                    if frame_data["status"] != "ok":
                        await websocket.send_json({
                            "type": "status_update",
                            "status": frame_data["status"]
                        })
                        continue
                
                # Process hands data and get update
                update = state_tracker.add_frame_data(frame_data["hands"])
                
                # Only send update if we have new data
                if update is not None:
                    await websocket.send_json(update)
                
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected after {frame_count} frames")
                break
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Internal server error",
                        "hands": []
                    })
                except:
                    break
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info(f"Connection closed. Frames processed: {frame_count}")