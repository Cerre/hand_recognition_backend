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
from hand_recognition import HandDetector, GestureAnalyzer

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

# Initialize hand detection components
detector = HandDetector(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
gesture_analyzer = GestureAnalyzer()

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
        
        # Use HandDetector to find hands
        frame, hands_data = detector.find_hands(frame_rgb)
        
        # Process each hand with GestureAnalyzer
        processed_hands = []
        for hand_data in hands_data:
            analysis = gesture_analyzer.analyze_hand(hand_data)
            processed_hands.append({
                'handedness': analysis['handedness'],
                'finger_count': analysis['finger_count']
            })

        return {"hands": processed_hands}

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