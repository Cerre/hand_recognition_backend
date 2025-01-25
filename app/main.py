from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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

@app.get("/socket.io/{path_params:path}")
async def handle_socketio(request: Request):
    """Handle Socket.IO polling requests to prevent 404 logs"""
    return {"message": "Socket.IO not supported. Please use WebSocket connection."}, 400

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
        return {"error": str(e), "hands": []}

@app.get("/")
async def root():
    logger.info("Health check endpoint accessed")
    return {"message": "Hand recognition backend is running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)
    authenticated = False
    logger.info(f"New client {client_id} attempting to connect")
    
    try:
        await websocket.accept()
        logger.info(f"Client {client_id} connected, awaiting authentication")
        
        while True:
            try:
                # Read the next message
                message = await websocket.receive()
                
                # Handle different message types
                if message["type"] == "websocket.disconnect":
                    logger.warning(f"Client {client_id} disconnected")
                    break
                
                if not authenticated:
                    if message["type"] != "websocket.receive":
                        continue
                        
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") != "auth":
                            await websocket.send_json({
                                "type": "error",
                                "message": "Authentication required. Send a message with type 'auth' and your token."
                            })
                            await websocket.close(code=4001)
                            return
                            
                        if data.get("token") != API_KEY:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Invalid authentication token"
                            })
                            await websocket.close(code=4001)
                            return
                            
                        authenticated = True
                        await websocket.send_json({
                            "type": "auth",
                            "status": "success",
                            "message": "Successfully authenticated"
                        })
                        logger.info(f"Client {client_id} authenticated successfully")
                        continue
                        
                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON format in authentication message"
                        })
                        await websocket.close(code=4001)
                        return
                
                else:
                    # Handle frame data after authentication
                    if message["type"] != "websocket.receive":
                        continue
                        
                    data = message["text"]
                    if ',' not in data:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid frame format. Expected base64 image data.",
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
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid JSON format: {str(e)}",
                    "hands": []
                })
            
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                logger.error(error_msg, exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": error_msg,
                        "hands": []
                    })
                except:
                    logger.error(f"Failed to send error message to client {client_id}")
                    break
                    
    except Exception as e:
        error_msg = f"WebSocket connection error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": error_msg
            })
        except:
            pass
        
    finally:
        logger.info(f"WebSocket connection closed for client {client_id}")