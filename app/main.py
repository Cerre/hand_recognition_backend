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
        
        logger.debug(f"Processed frame data - Left hand: {left_hand}, Right hand: {right_hand}")
        
        # Return immediate update
        return {
            "type": "score_update",
            "player": left_hand,   # Player number from left hand
            "points": right_hand,  # Points from right hand
            "timestamp": time()
        }

def count_fingers(landmarks) -> dict:
    """
    Count extended fingers using MediaPipe hand landmarks.
    Checks thumb angle relative to index finger.
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
        """Check if thumb is extended by comparing to index finger"""
        # Get relevant points
        thumb_tip = points[4]
        thumb_base = points[2]
        index_base = points[5]
        
        # Vector from thumb base to tip
        thumb_vector = thumb_tip - thumb_base
        # Vector along index finger base
        index_vector = index_base - thumb_base
        
        # Calculate angle between thumb and index finger
        angle = get_angle_between_vectors(thumb_vector, index_vector)
        
        # Thumb is extended if it's at a significant angle from the index finger
        return angle > 45
    
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
    logger.debug(f"Detected {count} extended fingers - Extended status: {fingers_extended}")
    
    return {
        "total_count": int(count)
    }
def count_fingers_by_distance(landmarks, handedness) -> dict:
    """
    Count extended fingers using distances from palm center.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        handedness: MediaPipe handedness classification
        
    Returns:
        dict: Contains required hand info and additional debug/visualization data
    """
    points = np.array([[l.x, l.y, l.z] for l in landmarks])
    
    # Calculate palm center using wrist and MCP joints
    palm_points = points[[0, 5, 9, 13, 17]]  # Wrist and all MCP joints
    palm_center = np.mean(palm_points, axis=0)
    
    # Get reference length (palm width) for normalization
    palm_width = np.linalg.norm(points[5] - points[17])  # Distance between index and pinky MCP
    
    def is_finger_extended(tip_idx: int, threshold: float = 1) -> bool:
        """Check if finger is extended based on normalized distance from palm"""
        distance = np.linalg.norm(points[tip_idx] - palm_center)
        normalized_distance = distance / palm_width
        return normalized_distance > threshold
    
    # Check each fingertip
    fingers_extended = [
        is_finger_extended(4, 0.8),    # Thumb (lower threshold)
        is_finger_extended(8),         # Index 
        is_finger_extended(12),        # Middle
        is_finger_extended(16),        # Ring
        is_finger_extended(20)         # Pinky
    ]
    
    count = sum(fingers_extended)
    
    # Required hand info
    hand_info = {
        'handedness': str(handedness.classification[0].label),
        'finger_count': int(count)
    }
    
    # Additional data for debugging/visualization
    debug_data = {
        'palm_center': palm_center.tolist(),
        'reference_length': float(palm_width),
        'fingers_extended': fingers_extended,
        'normalized_distances': [
            float(np.linalg.norm(points[tip_idx] - palm_center) / palm_width)
            for tip_idx in [4, 8, 12, 16, 20]
        ]
    }
    
    # Combine both dictionaries
    return {**hand_info, **debug_data}

def process_frame(base64_frame: str) -> Dict[str, Any]:
    """Process a single frame and detect hands"""
    try:
        # Check for empty or invalid frame data
        if not base64_frame or base64_frame == "data:,":
            logger.debug("Received empty frame during camera initialization")
            return {"hands": []}

        # Log frame size for debugging
        frame_size = len(base64_frame)
        logger.debug(f"Received frame of size: {frame_size} bytes")

        # Validate base64 format
        if ',' not in base64_frame or ';base64,' not in base64_frame:
            logger.debug("Received invalid base64 format during camera initialization")
            return {"hands": []}

        # Decode base64 image
        try:
            img_data = base64.b64decode(base64_frame.split(',')[1])
            if not img_data:
                logger.debug("Decoded image data is empty")
                return {"hands": []}
            logger.debug(f"Successfully decoded base64 image, size: {len(img_data)} bytes")
        except Exception as e:
            logger.debug(f"Base64 decoding failed during camera initialization: {str(e)}")
            return {"hands": []}

        # Check for empty image data
        if len(img_data) < 100:  # Arbitrary small size threshold for invalid frames
            logger.debug("Image data too small to be valid frame")
            return {"hands": []}

        nparr = np.frombuffer(img_data, np.uint8)
        if nparr.size == 0:
            logger.debug("Empty numpy array during camera initialization")
            return {"hands": []}

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.debug("Failed to decode image during camera initialization")
            return {"hands": []}

        # Log frame dimensions
        logger.debug(f"Frame dimensions: {frame.shape}")

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Process detected hands
        hand_data = []
        if results.multi_hand_landmarks:
            logger.debug(f"Detected {len(results.multi_hand_landmarks)} hands in frame")
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get finger data using new distance-based method
                finger_results = count_fingers_by_distance(hand_landmarks.landmark, handedness)
                
                # Extract required hand info
                hand_info = {
                    'handedness': finger_results['handedness'],
                    'finger_count': finger_results['finger_count'],
                    'palm_center': finger_results['palm_center'],
                    'reference_length': finger_results['reference_length'],
                    'fingers_extended': finger_results['fingers_extended'],
                    'normalized_distances': finger_results['normalized_distances']
                }
                
                logger.debug(f"Processed hand: {hand_info}")
                hand_data.append(hand_info)
        else:
            logger.debug("No hands detected in frame")

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
    logger.info(f"New client {client_id} attempting to connect from {websocket.client.host}")
    
    state_tracker = GestureStateTracker()
    frame_count = 0
    last_frame_time = time()
    
    try:
        await websocket.accept()
        logger.info(f"Client {client_id} connected successfully")
        
        # Send initial connection success message
        try:
            await websocket.send_json({"status": "connected"})
        except Exception as e:
            logger.error(f"Failed to send initial connection message to client {client_id}: {str(e)}")
            return
        
        while True:
            try:
                # Add timeout logging
                current_time = time()
                if current_time - last_frame_time > 5:  # Log if more than 5 seconds between frames
                    logger.warning(f"Long delay between frames for client {client_id}: {current_time - last_frame_time:.2f} seconds")
                
                frame_count += 1
                logger.debug(f"Processing frame {frame_count} for client {client_id}")
                
                try:
                    data = await websocket.receive_text()
                    last_frame_time = time()
                except WebSocketDisconnect as e:
                    logger.info(f"Client {client_id} disconnected normally while receiving frame {frame_count}")
                    break
                except Exception as e:
                    logger.error(f"Error receiving frame from client {client_id}: {str(e)}")
                    break
                
                if not data:
                    logger.warning(f"Empty frame received from client {client_id}")
                    continue
                
                if ',' not in data:
                    logger.warning(f"Client {client_id} sent invalid data format")
                    await websocket.send_json({
                        "error": "Invalid data format",
                        "hands": []
                    })
                    continue
                
                # Process frame and send immediate update
                frame_data = process_frame(data)
                logger.debug(f"Frame {frame_count} processing results: {frame_data}")
                
                update = state_tracker.add_frame_data(frame_data["hands"])
                logger.debug(f"Sending update to client {client_id}: {update}")
                
                try:
                    await websocket.send_json(update)
                except Exception as e:
                    logger.error(f"Failed to send update to client {client_id}: {str(e)}")
                    break
                
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected normally after processing {frame_count} frames")
                break
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_count} for client {client_id}: {str(e)}", exc_info=True)
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
        logger.info(f"WebSocket connection closed for client {client_id}. Total frames processed: {frame_count}")