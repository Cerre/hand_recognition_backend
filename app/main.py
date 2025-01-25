from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np
import base64
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing. In production, use your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

@app.get("/")
async def root():
    return {"message": "Hand recognition backend is running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("Client attempting to connect")
    await websocket.accept()
    logger.info("Client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            logger.info("Received frame")
            
            try:
                # Process the frame
                if ',' not in data:
                    logger.error("Invalid data format")
                    continue
                    
                img_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.error("Could not decode image")
                    continue
                
                # Process hands
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                hand_data = []
                if results.multi_hand_landmarks:
                    logger.info(f"Detected {len(results.multi_hand_landmarks)} hands")
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.append({
                                'x': float(landmark.x),
                                'y': float(landmark.y),
                                'z': float(landmark.z)
                            })
                        hand_data.append(landmarks)
                
                await websocket.send_json({
                    "hands": hand_data
                })
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                await websocket.send_json({
                    "hands": []
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("Connection closed")