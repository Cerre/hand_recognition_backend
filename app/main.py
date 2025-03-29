from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Security, Depends
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
from fastapi import status

# Import hand_recognition package
from hand_recognition import HandDetector
from tools.xgboost_predictor import xgboost_method

# Set up logging - Change to ERROR to only show critical issues
logging.basicConfig(
    level=logging.ERROR,  # Changed from WARNING to ERROR - will only show critical issues
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get API key from environment
API_TOKEN = os.getenv('API_KEY')
if not API_TOKEN:
    logger.error("No API_TOKEN environment variable set")  # Changed to error since it's critical

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
    static_image_mode=False,  # Keep tracking mode
    max_num_hands=2,
    min_detection_confidence=0.7,  # Restore to original value
    min_tracking_confidence=0.7    # Restore to original value
)

# Security setup
api_token_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Dependency function to validate API Key
async def get_api_key(api_key_header: str = Security(api_token_header)):
    if not API_TOKEN:
        logger.critical("API_TOKEN environment variable not set. Authentication disabled.")
        # In a real scenario, you might want to deny all connections if the token isn't set
        # raise HTTPException(status_code=503, detail="Server configuration error: API Token not set")
        # For now, let's allow connection but log critical error
        return None # Or some indicator that auth is effectively off

    if api_key_header == API_TOKEN:
        return api_key_header
    else:
        logger.warning(f"Authentication failed: Invalid API Key received.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

# Rate limiting setup
connection_attempts = defaultdict(list)  # IP -> list of timestamps
MAX_ATTEMPTS = 5  # Maximum connection attempts
RATE_LIMIT_WINDOW = 60  # Time window in seconds
RATE_LIMIT_CLEANUP_INTERVAL = 300  # Cleanup old entries every 5 minutes
last_cleanup = time()

# Performance monitoring
frame_metrics = {
    'processed_count': 0,
    'processing_times': [],
    'last_processed_time': 0,
    'detection_success_count': 0,
    'detection_fail_count': 0,
    'tracking_stats': {
        'both_hands': 0,
        'single_hand': 0,
        'no_hands': 0
    }
}

def create_connection_token(timestamp: str) -> str:
    """Create a secure connection token using HMAC"""
    if not API_TOKEN:
        return ""
    return hmac.new(
        API_TOKEN.encode(),
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

# Remove the logging functions
def update_metrics(processing_time: float, hands_count: int):
    """Update metrics without logging"""
    frame_metrics['processing_times'].append(processing_time)
    if len(frame_metrics['processing_times']) > 100:
        frame_metrics['processing_times'] = frame_metrics['processing_times'][-100:]
    
    # Update tracking stats
    if hands_count == 2:
        frame_metrics['tracking_stats']['both_hands'] += 1
    elif hands_count == 1:
        frame_metrics['tracking_stats']['single_hand'] += 1
    else:
        frame_metrics['tracking_stats']['no_hands'] += 1

def log_frame_debug(client_id: str, stage: str, details: Dict[str, Any]):
    """Helper function to log frame processing details - only logs at DEBUG level"""
    logger.debug(f"Client {client_id} - {stage}: {json.dumps(details, default=str)}")  # Changed to debug level

def normalize_brightness(image: np.ndarray, target_brightness: float = 127.0, max_adjustment: float = 0.7) -> np.ndarray:
    """Normalize image brightness while preserving detail and preventing over-adjustment.
    
    Args:
        image: Input BGR image
        target_brightness: Target mean brightness (default: 127.0 - middle of 0-255 range)
        max_adjustment: Maximum allowed adjustment factor to prevent extreme changes
    
    Returns:
        Normalized image
    """
    current_brightness = np.mean(image)
    
    # Calculate adjustment needed
    adjustment = target_brightness / current_brightness if current_brightness > 0 else 1.0
    
    # Limit adjustment range
    adjustment = max(1.0 - max_adjustment, min(1.0 + max_adjustment, adjustment))
    
    # Apply adjustment while preserving detail
    normalized = cv2.convertScaleAbs(image, alpha=adjustment, beta=0)
    return normalized

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better hand detection.
    
    Args:
        image: Input BGR image
    
    Returns:
        Enhanced image
    """
    # First normalize brightness
    image = normalize_brightness(image)
    
    # Apply adaptive histogram equalization to improve contrast while preserving local details
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def process_frame(base64_frame: str, client_id: str = "unknown") -> Dict[str, Any]:
    """Process a single frame and detect hands"""
    start_time = time()
    frame_metrics['processed_count'] += 1
    
    try:
        # Check if this is an auth message
        try:
            json_data = json.loads(base64_frame)
            if isinstance(json_data, dict) and 'type' in json_data:
                log_frame_debug(client_id, "Received JSON Message", {
                    "message_type": json_data.get('type'),
                    "length": len(base64_frame)
                })
                return {"hands": [], "status": "json_message"}
        except json.JSONDecodeError:
            pass  # Not a JSON message, continue with image processing

        # Handle empty frames during initialization more gracefully
        if not base64_frame:
            log_frame_debug(client_id, "Empty Frame", {"type": "empty"})
            frame_metrics['detection_fail_count'] += 1
            frame_metrics['tracking_stats']['no_hands'] += 1
            return {"hands": [], "status": "waiting_for_camera"}
            
        if base64_frame == "data:,":
            log_frame_debug(client_id, "Initializing Frame", {"type": "data:,"})
            frame_metrics['detection_fail_count'] += 1
            frame_metrics['tracking_stats']['no_hands'] += 1
            return {"hands": [], "status": "camera_initializing"}

        # Log frame format and size
        frame_format = base64_frame.split(',')[0] if ',' in base64_frame else "invalid"
        frame_size = len(base64_frame)
        log_frame_debug(client_id, "Frame Format", {
            "format": frame_format,
            "length": frame_size,
            "is_valid_base64": frame_format.startswith("data:image")
        })

        # Quick validation of base64 data
        if ',' not in base64_frame:
            log_frame_debug(client_id, "Invalid Frame", {
                "error": "no comma in base64",
                "received_start": base64_frame[:50] + "..." if len(base64_frame) > 50 else base64_frame
            })
            frame_metrics['detection_fail_count'] += 1
            frame_metrics['tracking_stats']['no_hands'] += 1
            return {"hands": [], "status": "invalid_frame"}

        # Decode base64 image
        try:
            img_data = base64.b64decode(base64_frame.split(',')[1])
            if not img_data:
                log_frame_debug(client_id, "Empty Image Data", {"error": "empty after decode"})
                frame_metrics['detection_fail_count'] += 1
                frame_metrics['tracking_stats']['no_hands'] += 1
                return {"hands": [], "status": "empty_frame"}
            
            log_frame_debug(client_id, "Image Data", {
                "decoded_size": len(img_data),
                "compression_ratio": len(img_data) / frame_size if frame_size > 0 else 0
            })

        except Exception as e:
            log_frame_debug(client_id, "Base64 Decode Error", {
                "error": str(e),
                "frame_start": base64_frame[:50] + "..." if len(base64_frame) > 50 else base64_frame
            })
            frame_metrics['detection_fail_count'] += 1
            frame_metrics['tracking_stats']['no_hands'] += 1
            return {"hands": [], "status": "decode_error"}

        # Convert to numpy array and decode image
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            log_frame_debug(client_id, "Image Decode Error", {
                "error": "cv2.imdecode returned None",
                "array_size": len(nparr),
                "array_min": int(nparr.min()),
                "array_max": int(nparr.max())
            })
            frame_metrics['detection_fail_count'] += 1
            frame_metrics['tracking_stats']['no_hands'] += 1
            return {"hands": [], "status": "invalid_image"}

        # Log original frame dimensions and basic image statistics
        original_brightness = np.mean(frame)
        original_std = np.std(frame)
        
        # Enhance image quality
        frame = enhance_image_quality(frame)
        
        # Log enhanced frame statistics
        enhanced_brightness = np.mean(frame)
        enhanced_std = np.std(frame)
        
        log_frame_debug(client_id, "Frame Analysis", {
            "width": frame.shape[1],
            "height": frame.shape[0],
            "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
            "original_brightness": float(original_brightness),
            "original_std": float(original_std),
            "enhanced_brightness": float(enhanced_brightness),
            "enhanced_std": float(enhanced_std),
            "brightness_adjustment": float(enhanced_brightness - original_brightness)
        })

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use HandDetector to find hands
        try:
            frame, hands_data = detector.find_hands(frame_rgb)
            detection_time = (time() - start_time) * 1000
            
            # Log detailed hand detection results
            hand_details = []
            for hand in hands_data:
                hand_details.append({
                    "handedness": hand.get('handedness', 'unknown'),
                    "confidence": float(hand.get('confidence', 0)),
                    "has_landmarks": 'landmarks' in hand,
                    "landmark_count": len(hand['landmarks']) if 'landmarks' in hand else 0
                })
            
            log_frame_debug(client_id, "Hand Detection Details", {
                "hands_found": len(hands_data),
                "detection_time_ms": detection_time,
                "hand_details": hand_details
            })

        except Exception as e:
            log_frame_debug(client_id, "Hand Detection Error", {
                "error": str(e),
                "traceback": str(e.__traceback__)
            })
            frame_metrics['detection_fail_count'] += 1
            frame_metrics['tracking_stats']['no_hands'] += 1
            return {"hands": [], "status": "detection_error"}
        
        # Process each hand with XGBoost method
        processed_hands = []
        for hand_data in hands_data:
            try:
                finger_count = count_fingers(hand_data['landmarks'])
                processed_hands.append({
                    'handedness': hand_data['handedness'],
                    'finger_count': finger_count,
                    'confidence': float(hand_data.get('confidence', 0))
                })
            except Exception as e:
                log_frame_debug(client_id, "Finger Count Error", {
                    "error": str(e),
                    "handedness": hand_data.get('handedness', 'unknown'),
                    "has_landmarks": 'landmarks' in hand_data
                })

        # Update success metrics
        frame_metrics['detection_success_count'] += 1
        processing_time = (time() - start_time) * 1000  # Convert to ms
        frame_metrics['last_processed_time'] = time()
        
        # Update metrics
        update_metrics(processing_time, len(processed_hands))

        # Log final results with more details
        log_frame_debug(client_id, "Processing Complete", {
            "hands_processed": len(processed_hands),
            "total_time_ms": processing_time,
            "status": "ok",
            "processed_hands": processed_hands
        })

        return {
            "hands": processed_hands, 
            "status": "ok",
            "processing_time_ms": processing_time
        }

    except Exception as e:
        log_frame_debug(client_id, "Unexpected Error", {
            "error": str(e),
            "traceback": str(e.__traceback__)
        })
        frame_metrics['detection_fail_count'] += 1
        frame_metrics['tracking_stats']['no_hands'] += 1
        return {"hands": [], "status": "error"}

# Remove the periodic logging setup
@app.on_event("startup")
async def startup_event():
    pass  # Remove the periodic logging task

@app.get("/")
async def root():
    logger.debug("Health check endpoint accessed")  # Changed to debug
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
async def websocket_endpoint(websocket: WebSocket, api_key: str = Security(get_api_key)):
    client_id = id(websocket)
    client_ip = websocket.client.host
    
    # Log client connection details at debug level - Now includes successful auth implicitly
    logger.info(f"Authenticated connection - Client ID: {client_id}, IP: {client_ip}")
    
    try:
        await websocket.accept()
        state_tracker = GestureStateTracker()
        frame_count = 0
        
        while True:
            try:
                data = await websocket.receive_text()
                # Log the received data (first 100 chars for brevity)
                logger.info(f"Client {client_id} received data (first 100 chars): {data[:100]}...")

                frame_data = process_frame(data, str(client_id))
                update = state_tracker.add_frame_data(frame_data["hands"])
                
                if update is not None:
                    await websocket.send_json(update)
                    
            except WebSocketDisconnect:
                logger.debug(f"Client {client_id} disconnected after {frame_count} frames")
                break
            except Exception as e:
                # Log the full exception traceback
                logger.exception(f"Detailed error processing frame for client {client_id}:")
                # Keep the original simple error log as well, if desired
                # logger.error(f"Error processing frame for client {client_id}: {str(e)}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Internal server error",
                        "hands": []
                    })
                except:
                    break
    except Exception as e:
        logger.error(f"Failed to establish WebSocket connection for client {client_id} from {client_ip}: {str(e)}")
        # Connection might already be closed by FastAPI due to auth HTTPException
        # No explicit close needed here unless accept() itself failed differently.