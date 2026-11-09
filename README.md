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
  - `hands_detect.py`: Hand detection implementation
  - `resnet-inference.py`: Contain a wrapper for
- **ml/src/models/**: Pre-trained models
  - `hand_landmarker.task`: MediaPipe hand landmarker model
- **backend/**: FastAPI prediction-ingestion endpoint and WebSocket stream
- **frontend/**: Frontend React UI

## Saving and using the custom CNN

After training `model` in `ml/src/cnn/network.ipynb`, run the final notebook cell to save the complete trained network to `ml/src/cnn/sign_language_cnn_25_epochs.pkl`. The checkpoint includes the layer structure and learned parameters.

Load it once and reuse it for predictions:

```python
from ml.src.cnn.inference import load_model, predict_image

model = load_model()
prediction = predict_image("path/to/hand-image.jpg", model=model)
print(prediction)  # {"letter": "A", "confidence": 0.95}
```

`predict_image` and `predict_bgr` use the training input format: OpenCV BGR, resize to 64×64, scale to `[0, 1]`, then convert to CHW. Only load checkpoints created by this project: Python pickle files can execute code when loaded.

## Development

All Python development should be done within the activated virtual environment. VS Code is configured to automatically use the virtual environment interpreter.

## Running the API and camera preview

The browser owns the webcam. Start the API and frontend, allow camera access in the browser, then select **Start live recognition** to send frames to the API.

### Windows

Start the frontend and API with the setup script. The frontend opens in a separate command window, and the current window runs Uvicorn:

```bat
setup.bat
```

The camera preview opens when the frontend loads. Recognition frames are JPEG binary messages sent to `/ws/recognition` websocket endpoint. The backend loads the YOLO detector and the CNN Classifier once on startup, then reuses the existing flip, hand-crop, and classifier preprocessing pipeline for each frame. Frames are captured at up to 60 attempts per second by default (`VITE_FRAME_RATE` can lower or raise the rate up to 60); the browser allows only one frame in flight and skips captures while waiting for a prediction, so actual inference frequency follows backend capacity.

Press **Pause** to stop sending frames and close the recognition WebSocket. Closing the page releases the browser camera.

### API data flow

- Recognition frontend clients connect to `ws://127.0.0.1:8000/ws/recognition`; each JPEG frame receives a JSON response containing the sign label, confidence, and status (`success`, `no_hand`, or `error`).
- `ws://127.0.0.1:8000/ws` remains available for clients consuming predictions posted to `POST /prediction`. That legacy endpoint validates and stores the latest prediction in the API process.
- The ingestion route has no authentication and is intended for local development. Do not expose it to an untrusted network without adding authentication and other deployment protections.

The interactive API documentation is available at `http://127.0.0.1:8000/docs`.
