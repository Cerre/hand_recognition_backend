# Hand Recognition Backend

## Project Overview
This is a real-time hand recognition and gesture tracking backend service built with FastAPI and MediaPipe. The system processes video streams to detect and analyze hand gestures, providing real-time feedback through WebSocket connections.

## Architecture

### Tech Stack
- **FastAPI**: Web framework for building the API endpoints and WebSocket server
- **MediaPipe**: Machine learning framework for hand detection and landmark tracking
- **OpenCV**: Computer vision library for image processing
- **WebSocket**: Real-time bidirectional communication protocol
- **Docker**: Containerization for deployment

### Core Components

#### 1. FastAPI Application (`app/main.py`)
- Main application entry point
- WebSocket endpoint for real-time communication
- CORS middleware configuration for frontend integration
- Health check endpoint

#### 2. Hand Detection and Processing
- Uses MediaPipe Hands for accurate hand landmark detection
- Supports detection of up to 2 hands simultaneously
- Configurable detection and tracking confidence thresholds
- Real-time frame processing and analysis

#### 3. Gesture Analysis
- `GestureStateTracker` class for gesture state management
- Finger counting algorithms:
  - Distance-based finger detection
  - Angle-based gesture recognition
- Support for both left and right hand tracking

## Deployment

### Infrastructure
- Containerized deployment using Docker
- Procfile configuration for platform deployment
- Environment variable support for configuration

### Endpoints
- `/`: Health check endpoint
- `/ws`: WebSocket endpoint for real-time hand tracking

## Dependencies
Key dependencies include:
- FastAPI and Uvicorn for API server
- MediaPipe for hand detection
- OpenCV for image processing
- NumPy for numerical computations
- Additional utilities for WebSocket handling and data processing

## Frontend Integration
- CORS configured for specific origins:
  - Local development: `http://localhost:3000`
  - Production: `https://hand-tracker-web.vercel.app`
- WebSocket-based real-time communication
- Base64 encoded frame processing

## Performance Considerations
- Optimized for real-time processing
- Configurable detection confidence thresholds
- Efficient frame processing pipeline
- Minimal state management for improved performance

## Security
- CORS protection
- WebSocket connection management
- Input validation and sanitization

## Future Improvements
- Additional gesture recognition patterns
- Performance optimizations for high-load scenarios
- Enhanced error handling and recovery
- Extended analytics and tracking capabilities 