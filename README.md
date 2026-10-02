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
- **backend/**: Backend API (coming soon)
- **frontend/**: Frontend UI (coming soon)

## Development

All Python development should be done within the activated virtual environment. VS Code is configured to automatically use the virtual environment interpreter.
