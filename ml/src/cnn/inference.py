"""Load and run inference with the custom sign-language CNN checkpoint."""

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

CNN_DIRECTORY = Path(__file__).resolve().parent
SOURCE_DIRECTORY = CNN_DIRECTORY.parent
for module_directory in (SOURCE_DIRECTORY, CNN_DIRECTORY):
    if str(module_directory) not in sys.path:
        sys.path.insert(0, str(module_directory))

from data_loader import CLASS_NAMES
from network import Network

DEFAULT_MODEL_PATH = CNN_DIRECTORY / "sign_language_cnn_25_epochs.pkl"
INPUT_SIZE = 64


def load_model(checkpoint_path=DEFAULT_MODEL_PATH):
    """Load a trusted pickle checkpoint produced by the CNN training notebook."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

    with path.open("rb") as checkpoint_file:
        model = pickle.load(checkpoint_file)

    if not isinstance(model, Network):
        raise TypeError(f"Checkpoint does not contain a {Network.__name__} model")
    return model


def predict_bgr(image_bgr, model):
    """Predict one OpenCV BGR image using the notebook's training preprocessing."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot predict from an empty image")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image")

    image = cv2.resize(
        image_bgr,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    image = image.astype(np.float32, copy=False) / np.float32(255.0)
    image_chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    scores = np.asarray(model.predict(model.layers, image_chw), dtype=np.float32).reshape(-1)

    if scores.shape != (len(CLASS_NAMES),) or not np.isfinite(scores).all():
        raise ValueError("Model returned invalid class scores")

    class_index = int(np.argmax(scores))
    return {
        "letter": CLASS_NAMES[class_index],
        "confidence": float(scores[class_index]),
    }


def predict_image(image_path, model=None):
    """Load an image from disk and return its predicted label and confidence."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    loaded_model = model if model is not None else load_model()
    return predict_bgr(image, loaded_model)
