# Real-Time-Sign-Language-Recognition-and-Translation

Developing a real-time sign language recognition and translation platform using Computer Vision

## Project Setup

This project uses a Python virtual environment to manage dependencies. Follow the steps below to set up the project.

### Prerequisites

- Python 3.8 or higher
- Git (for version control)

### Installation

#### Windows

1. Run the setup script:
   ```bash
   setup.bat
   ```
   This will create the virtual environment and install all dependencies.

#### macOS/Linux

1. Make the setup script executable:
   ```bash
   chmod +x setup.sh
   ```
2. Run the setup script:
   ```bash
   ./setup.sh
   ```

### Activating the Virtual Environment

#### Windows

```bash
.venv\Scripts\activate.bat
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### Deactivating the Virtual Environment

```bash
deactivate
```

### Installing Additional Dependencies

When the virtual environment is activated, install packages using:

```bash
pip install <package_name>
```

To save new dependencies to requirements.txt:

```bash
pip freeze > requirements.txt
```

## Project Structure

- **ml/src/cnn/**: CNN model implementation
  - `activation.py`: Activation functions
  - `layers.py`: Neural network layers
  - `losses.py`: Loss functions
  - `network.py`: Network architecture
- **ml/src/hand_detection/**: Hand detection module
  - `handsDetect.py`: Hand detection implementation
- **ml/src/models/**: Pre-trained models
  - `hand_landmarker.task`: MediaPipe hand landmarker model
- **backend/**: FastAPI prediction-ingestion endpoint and WebSocket stream
- **frontend/**: Frontend UI (coming soon)

## Development

All Python development should be done within the activated virtual environment. VS Code is configured to automatically use the virtual environment interpreter.

## Running the API and camera

The API and camera run as separate processes. Start the API first, then start the camera in a second terminal from the project root.

### Windows

Start the API (the setup script installs dependencies and then starts Uvicorn):

```bat
setup.bat
```

Or, if dependencies are already installed:

```bat
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

In a second terminal, start the camera and point it to the API's prediction-ingestion route:

```bat
.venv\Scripts\python.exe -m ml.src.hand_detection.hand_detect --api-url http://127.0.0.1:8000/prediction
```

Press `x` in the camera window to stop the camera process. Stop Uvicorn with `Ctrl+C`.

### API data flow

- The camera process posts JSON predictions to `POST /prediction`, for example `{"letter":"A","confidence":0.95}`. The API responds with `{"status":"ok"}`; `letter` must be non-empty and `confidence` must be between `0.0` and `1.0` (invalid input receives HTTP 422). When no hand is detected, the camera posts `{"letter":"None","confidence":0.0}`.
- The API validates and stores the latest prediction in its own process.
- Frontend clients connect to `ws://127.0.0.1:8000/ws`. The WebSocket sends the current prediction immediately and sends another message when the letter or confidence changes.
- `POST /prediction` is an ingestion endpoint for the camera, not a frontend polling route. The camera publisher keeps at most one pending update and sends it on a background thread, so HTTP requests do not block camera frame capture. If the API is unavailable, the camera keeps running and logs a warning; it will send later updates when the API is reachable again.
- The ingestion route has no authentication and is intended for local development. Do not expose it to an untrusted network without adding authentication and other deployment protections.

The interactive API documentation is available at `http://127.0.0.1:8000/docs`.
