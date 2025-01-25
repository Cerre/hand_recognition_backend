from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import base64
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hand recognition backend is running"}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("Client attempting to connect")
    await websocket.accept()
    logger.info("Client connected")
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"Received frame data")
                
                # Extract and decode base64 image
                img_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                    
                # Get image dimensions
                height, width = frame.shape[:2]
                
                # Process hands
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                hand_data = []
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Get bounding box
                        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                        bbox = {
                            'x_min': min(x_coords),
                            'x_max': max(x_coords),
                            'y_min': min(y_coords),
                            'y_max': max(y_coords),
                        }
                        
                        # Add landmarks and bounding box
                        hand_info = {
                            'landmarks': [
                                {
                                    'x': float(landmark.x),
                                    'y': float(landmark.y),
                                    'z': float(landmark.z)
                                } for landmark in hand_landmarks.landmark
                            ],
                            'bbox': bbox
                        }
                        hand_data.append(hand_info)
                
                await websocket.send_json({
                    "hands": hand_data,
                    "image_dims": {"width": width, "height": height}
                })
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                await websocket.send_json({"hands": [], "image_dims": None})
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")