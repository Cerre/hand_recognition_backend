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
    level=logging.DEBUG,  # Revert to INFO for production
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
    min_detection_confidence=0.5,  # Temporarily lower from 0.7
    min_tracking_confidence=0.5    # Temporarily lower from 0.7
)

# Security setup
# Remove the APIKeyHeader instance as it's not directly used in the dependency anymore
# api_token_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Dependency function to validate API Key
# Change signature to accept WebSocket object
async def get_api_key(websocket: WebSocket):
    # 1. Check if the server has the API_TOKEN configured
    if not API_TOKEN:
        logger.critical("CRITICAL SERVER ERROR: API_TOKEN environment variable not set.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server configuration error: Authentication unavailable."
        )

    # 2. Determine the source and value of the provided key
    api_key_header_value = websocket.headers.get("X-API-Key")
    token_query_param = websocket.query_params.get("token")

    provided_key = None
    source = None

    if api_key_header_value:
        provided_key = api_key_header_value
        source = "header"
    elif token_query_param:
        provided_key = token_query_param
        source = "query parameter"

    # 3. Check if any key was provided
    if provided_key is None:
        logger.warning(f"Authentication failed: Missing X-API-Key header or token query parameter.")
        raise HTTPException(
            status_code=status.WS_1008_POLICY_VIOLATION,
            detail="API Key required in X-API-Key header or token query parameter",
        )

    # 4. Compare the provided key with the server's key
    if hmac.compare_digest(provided_key, API_TOKEN):
        # Use compare_digest for security against timing attacks
        logger.debug(f"Authentication successful using {source}.") # Log success source
        return provided_key # Return the validated key
    else:
        logger.warning(f"Authentication failed: Invalid API Key received via {source}.")
        raise HTTPException(
            status_code=status.WS_1008_POLICY_VIOLATION,
            detail=f"Invalid API Key provided via {source}",
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

class GestureStateTracker:
    def __init__(self):
        self.last_update_time = time()
        # Allow updates slightly faster than 30fps to avoid skipping due to jitter
        self.min_update_interval = 1.0 / 35  # Allow up to ~35 FPS updates

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
        left_landmarks = []  # Add landmark lists
        right_landmarks = [] # Add landmark lists
        
        for hand in hands_data:
            if hand['handedness'] == 'Left':
                left_hand = hand['finger_count']
                # Convert ndarray to list for JSON serialization
                landmarks_array = hand.get('landmarks', [])
                left_landmarks = landmarks_array.tolist() if isinstance(landmarks_array, np.ndarray) else landmarks_array
            else:
                right_hand = hand['finger_count']
                # Convert ndarray to list for JSON serialization
                landmarks_array = hand.get('landmarks', [])
                right_landmarks = landmarks_array.tolist() if isinstance(landmarks_array, np.ndarray) else landmarks_array
        
        self.last_update_time = current_time
        
        # Return immediate update
        return {
            "type": "score_update",
            "player": right_hand,   # Use original key for right hand count
            "points": left_hand,    # Use original key for left hand count
            "left_landmarks": left_landmarks, # Include landmarks
            "right_landmarks": right_landmarks, # Include landmarks
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
    timings = {} # Dictionary to store timings

    try:
        t0 = time()
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
        t1 = time()
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
        timings['decode_base64_cv2'] = (time() - t1) * 1000

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
        t2 = time()
        frame = enhance_image_quality(frame)
        timings['enhance_quality'] = (time() - t2) * 1000
        
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
        t3 = time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timings['convert_rgb'] = (time() - t3) * 1000
        
        # Use HandDetector to find hands
        t4 = time() # Start timing for the block including find_hands
        try:
            # Call find_hands (drawing is disabled)
            _frame_ignored, hands_data = detector.find_hands(frame_rgb, draw=False)
            
            # Simplified logging (original detection_time included more than just find_hands)
            # Log detailed hand detection results (only if log level is DEBUG)
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
            
        # Log the time for the block containing find_hands and detail extraction
        timings['find_hands_block'] = (time() - t4) * 1000 
        
        # Process each hand with XGBoost method
        t5 = time()
        processed_hands = []
        finger_count_times = []
        for hand_data in hands_data:
            t_fc_start = time()
            try:
                # Call the new predict_count method once per hand
                finger_count = xgboost_method.predict_count(hand_data['landmarks'])
                processed_hands.append({
                    'handedness': hand_data['handedness'],
                    'finger_count': finger_count, # Use the predicted count
                    'confidence': float(hand_data.get('confidence', 0)),
                    'landmarks': hand_data.get('landmarks', [])
                })
            except Exception as e:
                log_frame_debug(client_id, "Finger Count Error", {
                    "error": str(e),
                    "handedness": hand_data.get('handedness', 'unknown'),
                    "has_landmarks": 'landmarks' in hand_data
                })
            finger_count_times.append((time() - t_fc_start) * 1000)
        timings['count_fingers_total'] = sum(finger_count_times)
        timings['count_fingers_per_hand_avg'] = sum(finger_count_times) / len(hands_data) if hands_data else 0
        timings['process_hands_loop'] = (time() - t5) * 1000

        # Update success metrics
        frame_metrics['detection_success_count'] += 1
        total_processing_time = (time() - start_time) * 1000
        timings['total_function_time'] = total_processing_time
        
        # Update metrics
        update_metrics(total_processing_time, len(processed_hands))

        # Log final results with timings
        log_frame_debug(client_id, "Processing Timings (ms)", timings)
        log_frame_debug(client_id, "Processing Complete", {
            "hands_processed": len(processed_hands),
            "total_time_ms": total_processing_time,
            "status": "ok",
            "processed_hands": processed_hands
        })

        return {
            "hands": processed_hands,
            "status": "ok",
            "processing_time_ms": total_processing_time
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
# Change from Security() to Depends()
async def websocket_endpoint(websocket: WebSocket, api_key: str = Depends(get_api_key)):
    client_id = id(websocket)
    client_ip = websocket.client.host
    
    logger.info(f"Authenticated connection - Client ID: {client_id}, IP: {client_ip}")
    
    is_processing = False # Flag to track if a frame is currently being processed
    
    try:
        await websocket.accept()
        state_tracker = GestureStateTracker()
        frame_count = 0
        
        while True:
            update_to_send = None # Store potential update
            try:
                data_bytes = await websocket.receive_bytes()

                # --- Frame Skipping Logic ---
                if is_processing:
                    logger.debug(f"Client {client_id} - Skipping frame, already processing.")
                    continue # Skip this frame if already busy
                # --------------------------
                
                try:
                    is_processing = True # Set flag before starting processing
                    
                    # Encode the received bytes as base64 and prepend the data URL prefix
                    data_base64 = f"data:image/jpeg;base64,{base64.b64encode(data_bytes).decode('utf-8')}"
                    logger.info(f"Client {client_id} received {len(data_bytes)} bytes. Starting processing.")
    
                    # Pass the base64 encoded string to process_frame
                    frame_data = process_frame(data_base64, str(client_id))
                    # Check if state_tracker allows an update based on its internal timing
                    update_to_send = state_tracker.add_frame_data(frame_data.get("hands", [])) 

                    # Send update if available
                    if update_to_send is not None:
                        await websocket.send_json(update_to_send)
                        logger.debug(f"Client {client_id} - Sent update: ...") # Keep log concise
                    else:
                        logger.debug(f"Client {client_id} - Update skipped by GestureStateTracker rate limit.")
                        
                finally:
                    is_processing = False # Reset flag after processing (and sending) is done
                    frame_count += 1 # Increment frame count only after processing attempt

            except WebSocketDisconnect:
                logger.debug(f"Client {client_id} disconnected after processing attempt on {frame_count} frames")
                break
            except Exception as e:
                logger.exception(f"Detailed error processing frame for client {client_id}:")
                is_processing = False # Ensure flag is reset even on error within processing block
                try:
                    # Attempt to send an error message, but don't block if it fails
                    await websocket.send_json({"type": "error", "message": "Internal server error"})
                except Exception:
                    logger.warning(f"Client {client_id} - Failed to send error message, connection might be closed.")
                    break # Assume connection is broken if we can't send error
    
    except Exception as e:
        logger.error(f"WebSocket connection failed or closed unexpectedly for client {client_id}: {str(e)}")