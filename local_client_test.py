import cv2
import websockets
import asyncio
import os
import logging
import time
from dotenv import load_dotenv
import json # Import json
from mediapipe.python.solutions import hands as mp_hands # Import hands solution for connections

# Load environment variables from .env file (or specify path)
load_dotenv(dotenv_path='.env')

# # --- REMOVE DEBUG PRINT --- #
# print("DEBUG: Attempting to load API_KEY...")
# api_key_check = os.getenv('API_KEY')
# if api_key_check:
#     print("DEBUG: API_KEY loaded.")
# else:
#     print("DEBUG: API_KEY **NOT** loaded.")
# # ------------------- #

# Add a shared variable for latest hand data (landmarks and finger counts)
latest_hand_data = {"left": {"landmarks": [], "fingers": None}, 
                    "right": {"landmarks": [], "fingers": None}}

# Add a check to see if the .env file was loaded
if os.getenv('API_KEY'):
    logging.info(".env file loaded successfully and API_KEY found.")
else:
    # Make sure this warning is indented under the else
    logging.warning(f"Failed to load API_KEY from {os.path.abspath('.env')}. Ensure the file exists and is formatted correctly.")

# --- Configuration --- #
WEBSOCKET_URL = "ws://localhost:8000/ws"
FRAME_RATE = 25  # Set stable frame rate for production
CAM_INDEX = 0    # Default webcam index
JPEG_QUALITY = 80 # Quality for JPEG encoding (0-100)
# ------------------- #

# Change level to INFO for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def receive_messages(websocket):
    """Listen for messages, parse landmarks and finger counts, and update the shared variable."""
    global latest_hand_data # Access the global variable
    try:
        async for message in websocket:
            # logging.info(f"< Received: {message}") # Optional: Reduce log spam
            try:
                data = json.loads(message)
                if data.get("type") == "score_update":
                    # Update landmarks and finger counts using original keys from server
                    latest_hand_data["left"]["landmarks"] = data.get("left_landmarks", [])
                    latest_hand_data["right"]["landmarks"] = data.get("right_landmarks", [])
                    latest_hand_data["left"]["fingers"] = data.get("points") # Use 'points' for left hand count
                    latest_hand_data["right"]["fingers"] = data.get("player") # Use 'player' for right hand count
                    
                    # Clear finger count if landmarks are empty (hand not detected)
                    if not latest_hand_data["left"]["landmarks"]:
                        latest_hand_data["left"]["fingers"] = None
                    if not latest_hand_data["right"]["landmarks"]:
                        latest_hand_data["right"]["fingers"] = None

            except json.JSONDecodeError:
                logging.warning(f"Received non-JSON message: {message}")
            except Exception as e:
                logging.exception(f"Error processing received message: {e}")

    except websockets.exceptions.ConnectionClosedOK:
        logging.info("Server closed the connection normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"Connection closed with error: {e}")
    except Exception as e:
        logging.exception(f"Error receiving messages: {e}")

async def send_frames():
    """Capture frames, draw landmarks and finger counts, and send frames over WebSocket."""
    global latest_hand_data # Access the global variable
    api_key = os.getenv('API_KEY')
    if not api_key:
        logging.error("API_KEY environment variable not set. Please set it and try again.")
        return

    uri = f"{WEBSOCKET_URL}?token={api_key}"
    logging.info(f"Attempting to connect to {uri}...")

    # # --- REMOVE DEBUG PRINT --- #
    # print(f"DEBUG: Using URI: {uri}")
    # print("DEBUG: Attempting WebSocket connect...")
    # # ------------------- #
    try:
        async with websockets.connect(uri) as websocket:
            # # --- REMOVE DEBUG PRINT --- #
            # print("DEBUG: WebSocket connected successfully.")
            # # ------------------- #
            logging.info("Connected to WebSocket server.")

            # Start the receiver task
            receiver_task = asyncio.create_task(receive_messages(websocket))
            
            # # --- REMOVE DEBUG PRINT --- #
            # print("DEBUG: Attempting to open camera...")
            # # ------------------- #
            cap = cv2.VideoCapture(CAM_INDEX)
            if not cap.isOpened():
                # # --- REMOVE DEBUG PRINT --- #
                # print(f"DEBUG: Failed to open camera {CAM_INDEX}.")
                # # ------------------- #
                logging.error(f"Error: Could not open camera with index {CAM_INDEX}.")
                receiver_task.cancel() # Stop receiver if camera fails
                return
            # # --- REMOVE DEBUG PRINT --- #
            # print(f"DEBUG: Camera {CAM_INDEX} opened.")
            # # ------------------- #

            logging.info(f"Camera {CAM_INDEX} opened successfully. Starting frame sending (Target FPS: {FRAME_RATE})..." )

            try:
                # # --- REMOVE DEBUG PRINT --- #
                # print("DEBUG: Entering main loop...")
                # # ------------------- #
                while True:
                    iter_start_time = time.time()

                    # --- Capture --- #
                    ret, frame = cap.read()
                    if not ret:
                        logging.warning("Could not read frame from camera.")
                        await asyncio.sleep(0.1) # Wait a bit before retrying
                        continue
                    # ------------- #

                    # --- Resize Frame --- #
                    target_width = 480
                    target_height = 360
                    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    # -------------------- #

                    # --- Encode --- #
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    result, encoded_image = cv2.imencode('.jpg', resized_frame, encode_param)
                    if not result:
                        logging.warning("Failed to encode frame to JPEG.")
                        continue
                    # ------------ #

                    # --- Display --- #
                    display_frame = frame.copy() # Display original resolution
                    display_frame = cv2.flip(display_frame, 1)
                    height, width, _ = display_frame.shape

                    # Draw latest landmarks and connections
                    for hand_type, hand_data in latest_hand_data.items():
                        landmarks = hand_data["landmarks"]
                        if not landmarks: 
                            continue
                        color = (0, 255, 0) if hand_type == 'right' else (255, 0, 0)
                        for connection in mp_hands.HAND_CONNECTIONS:
                            start_idx = connection[0]
                            end_idx = connection[1]
                            if start_idx < len(landmarks) and end_idx < len(landmarks):
                                start_point = landmarks[start_idx]
                                end_point = landmarks[end_idx]
                                if len(start_point) >= 2 and len(end_point) >= 2: 
                                    start_mirrored_x = 1.0 - start_point[0] 
                                    end_mirrored_x = 1.0 - end_point[0]
                                    start_px = (int(start_mirrored_x * width), int(start_point[1] * height))
                                    end_px = (int(end_mirrored_x * width), int(end_point[1] * height))
                                    cv2.line(display_frame, start_px, end_px, color, 2)
                        for i, lm in enumerate(landmarks):
                           if len(lm) >= 2:
                                mirrored_x = 1.0 - lm[0] 
                                cx, cy = int(mirrored_x * width), int(lm[1] * height)
                                cv2.circle(display_frame, (cx, cy), 5, color, cv2.FILLED)

                    # --- Display Finger Counts --- #
                    y_pos = 30 
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_color = (255, 255, 255)
                    line_type = 2
                    left_fingers = latest_hand_data["left"]["fingers"]
                    right_fingers = latest_hand_data["right"]["fingers"]
                    if latest_hand_data["left"]["landmarks"]:
                        text_left = f"Left Hand: {left_fingers if left_fingers is not None else 'N/A'} fingers"
                        cv2.putText(display_frame, text_left, (10, y_pos), font, font_scale, font_color, line_type)
                        y_pos += 30
                    if latest_hand_data["right"]["landmarks"]:
                         text_right = f"Right Hand: {right_fingers if right_fingers is not None else 'N/A'} fingers"
                         cv2.putText(display_frame, text_right, (10, y_pos), font, font_scale, font_color, line_type)
                    # ----------------------------- #

                    cv2.imshow('Local Client Camera View', display_frame)
                    key = cv2.waitKey(1) 
                    if key & 0xFF == ord('q'):
                        break
                    # ------------- #
                    
                    # --- Send --- #        
                    await websocket.send(encoded_image.tobytes())
                    # ---------- #

                    # --- Rate Limit Sleep --- #
                    processing_duration = time.time() - iter_start_time
                    target_interval = 1.0 / FRAME_RATE
                    sleep_duration = max(0, target_interval - processing_duration)
                    await asyncio.sleep(sleep_duration)
                    # ---------------------- #

                    # --- REMOVE Log Timings --- #
                    # total_iter_time = time.time() - iter_start_time
                    # print(
                    #     f"DEBUG Client Loop Timings (ms): "
                    #     f"Total={total_iter_time*1000:.1f} | "
                    #     # ... (removed detailed timings)
                    # )
                    # ----------------- #

            except websockets.exceptions.ConnectionClosed:
                logging.info("Connection closed by server.")
            except KeyboardInterrupt:
                logging.info("Interrupted by user.")
            finally:
                # Clean up
                cap.release()
                cv2.destroyAllWindows()
                logging.info("Camera released.")
                if not receiver_task.done():
                    receiver_task.cancel()
                    try:
                        await receiver_task
                    except asyncio.CancelledError:
                        logging.info("Receiver task cancelled.")
                logging.info("Client shutting down.")

    except websockets.exceptions.InvalidURI:
        logging.error(f"Invalid WebSocket URI: {uri}")
    except websockets.exceptions.ConnectionClosedError as e:
         logging.error(f"Connection failed: {e}. Is the server running at {WEBSOCKET_URL}?")
    except ConnectionRefusedError:
        logging.error(f"Connection refused. Ensure the server is running at {WEBSOCKET_URL}.")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(send_frames())
    except KeyboardInterrupt:
        logging.info("Client stopped.") 