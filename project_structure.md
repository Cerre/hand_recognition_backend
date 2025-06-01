# Project Structure

```
hand_recognition_backend/
├── app/                      # Production backend (existing)
│   └── main.py              # FastAPI application
├── hand_recognition/        # Core algorithm library
│   ├── __init__.py
│   ├── detector.py          # Hand detection and landmark extraction
│   ├── gesture.py           # Gesture analysis and finger counting
│   └── utils.py             # Shared utilities
├── tools/                   # Development and testing tools
│   ├── camera_test.py       # Local camera visualization
│   └── algorithm_debug.py   # Algorithm debugging utilities
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── Dockerfile              # Production container
└── Procfile               # Production deployment
```

## Component Responsibilities

### Core Library (`hand_recognition/`)
- Pure algorithm implementation
- No web/API dependencies
- Reusable across different applications
- Easy to test and debug

### Production Backend (`app/`)
- FastAPI application (existing)
- WebSocket handling
- Production endpoints
- Frontend integration

### Development Tools (`tools/`)
- Local camera visualization
- Algorithm testing
- Performance profiling
- Debug utilities

## Usage

### For Production
Continue using the existing deployment setup:
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### For Local Development
Use the camera test tool:
```bash
python local_client_test.py
```

This will:
1. Open your laptop camera
2. Show the video feed
3. Overlay hand landmarks
4. Display finger counting results
5. Allow algorithm parameter tuning 