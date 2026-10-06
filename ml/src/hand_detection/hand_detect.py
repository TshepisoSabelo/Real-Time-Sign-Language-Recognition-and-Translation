import pickle as plk
import json
import logging
import queue
import threading
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import argparse
from pathlib import Path

if __package__:
    from .resnet_inference import ResNetSignLanguageClassifier
else:
    from resnet_inference import ResNetSignLanguageClassifier
#==============================================================================
#Open Camera object
#xmodel = YOLO("yolo26n.pt")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_PATH = PROJECT_ROOT / "ml" / "src" / "models"
DEFAULT_DETECTOR = PROJECT_ROOT / "runs" / "pose" / "train-9" / "weights" / "best.pt"
DEFAULT_CHECKPOINT = DEFAULT_MODELS_PATH / "resnet_sign_language.pth"


def load_inference_models(detector_path=DEFAULT_DETECTOR, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
    """Load the detector and sign classifier once for a recognition session."""
    detection_model = YOLO(str(detector_path))
    classifier = ResNetSignLanguageClassifier(checkpoint_path=checkpoint_path, device=device)
    return detection_model, classifier


def process_frame(image_bgr, detection_model, classifier):
    """Run the existing hand detection, crop, and classification pipeline."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot process an empty camera frame")

    image = cv2.flip(image_bgr, 1)
    result = detection_model(image, verbose=False)[0]
    letter = "None"
    best_confidence = 0.0
    selected_hand = None

    for box, confidence in zip(result.boxes.xyxy, result.boxes.conf):
        confidence = float(confidence)
        if confidence < 0.5 or confidence <= best_confidence:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * 0.20)
        pad_y = int(height * 0.20)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image.shape[1], x2 + pad_x)
        y2 = min(image.shape[0], y2 + pad_y)

        hand = image[y1:y2, x1:x2]
        predictions = classifier.predict_bgr(hand, top_k=1)
        letter, best_confidence = predictions[0]
        selected_hand = hand

    prediction = {"letter": letter, "confidence": float(best_confidence)}
    return prediction, result, image, selected_hand


def predict_frame(image_bgr, detection_model, classifier):
    """Return the best sign prediction for one BGR camera frame."""
    prediction, _, _, _ = process_frame(image_bgr, detection_model, classifier)
    return prediction


def create_preview_frames(prediction, result, hand):
    """Create the annotated tracker image and classifier crop preview."""
    tracker_frame = result.plot()
    cv2.putText(
        tracker_frame,
        f"{prediction['letter']} {prediction['confidence']:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    if hand is None:
        hand_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(
            hand_frame,
            "No hand detected",
            (35, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    else:
        hand_frame = hand

    return tracker_frame, hand_frame


def process_frame_with_preview(image_bgr, detection_model, classifier):
    """Run recognition and prepare the associated OpenCV display frames."""
    prediction, result, _, hand = process_frame(
        image_bgr,
        detection_model,
        classifier,
    )
    tracker_frame, hand_frame = create_preview_frames(prediction, result, hand)
    return prediction, tracker_frame, hand_frame


class OpenCVPreview:
    """Display the latest recognition frames without owning the webcam."""

    def __init__(self):
        self.frames = queue.Queue(maxsize=1)
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return

            self.frames = queue.Queue(maxsize=1)
            self.thread = threading.Thread(
                target=self._display_frames,
                name="opencv-recognition-preview",
                daemon=True,
            )
            self.thread.start()

    def publish(self, tracker_frame, hand_frame):
        thread = self.thread
        if not thread or not thread.is_alive():
            return

        frames = (tracker_frame.copy(), hand_frame.copy())
        try:
            self.frames.put_nowait(frames)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(frames)

    def close(self):
        with self.lock:
            thread = self.thread
            if not thread:
                return

            while True:
                try:
                    self.frames.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass

        thread.join(timeout=2.0)
        with self.lock:
            if not thread.is_alive():
                self.thread = None

    def _display_frames(self):
        try:
            while True:
                frames = self.frames.get()
                if frames is None:
                    break

                tracker_frame, hand_frame = frames
                cv2.imshow("Hand Tracker", tracker_frame)
                cv2.imshow("Hand", hand_frame)
                cv2.waitKey(1)
        except cv2.error:
            logger.exception("Could not display the OpenCV recognition preview")
        finally:
            for window in ("Hand Tracker", "Hand"):
                try:
                    cv2.destroyWindow(window)
                except cv2.error:
                    logger.debug("OpenCV preview window was already closed: %s", window)


class PredictionPublisher:
    """Send the latest camera prediction to the API without blocking capture."""

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.pending = queue.Queue(maxsize=1)
        self.api_available = None
        self.worker = threading.Thread(target=self._send_pending, daemon=True)
        self.worker.start()

    def publish(self, prediction):
        try:
            self.pending.put_nowait(prediction)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait(prediction)

    def _send_pending(self):
        while True:
            prediction = self.pending.get()
            if prediction is None:
                return

            request = Request(
                self.endpoint,
                data=json.dumps(prediction).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=1.0) as response:
                    response.read()
                if self.api_available is False:
                    logger.info("Prediction API connection restored")
                self.api_available = True
            except (URLError, TimeoutError, OSError) as error:
                if self.api_available is not False:
                    logger.warning("Could not send prediction to API: %s", error)
                self.api_available = False

    def close(self):
        while True:
            try:
                self.pending.put_nowait(None)
                break
            except queue.Full:
                try:
                    self.pending.get_nowait()
                except queue.Empty:
                    pass
        self.worker.join(timeout=2.0)


def run_camera(
    detector_path=DEFAULT_DETECTOR,
    checkpoint_path=DEFAULT_CHECKPOINT,
    device=None,
    api_url="http://127.0.0.1:8000/prediction",
):
    """Runs hand detection followed by ResNet sign classification."""
    detection_model, classifier = load_inference_models(
        detector_path,
        checkpoint_path,
        device,
    )
    with open(DEFAULT_MODELS_PATH / "comparator_model", "rb") as f:
        comparator_model = plk.load(f)

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    publisher = PredictionPublisher(api_url)

    try:
        while True:
            success, image = capture.read()
            if not success:
                print("Unable to read a frame from the camera")
                break

            prediction, result, image, hand = process_frame(
                image,
                detection_model,
                classifier,
            )
            letter = prediction["letter"]
            best_confidence = prediction["confidence"]

            publisher.publish({
                "letter": letter,
                "confidence": float(best_confidence),
            })

            annotated, hand_preview = create_preview_frames(
                prediction,
                result,
                hand,
            )
            cv2.imshow("Hand", hand_preview)
            cv2.imshow("Hand Tracker", annotated)

            if cv2.waitKey(1) & 0xFF == ord("x"):
                break
    finally:
        publisher.close()
        capture.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run hand detection and ResNet sign prediction")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/prediction",
        help="Prediction ingestion URL of the running API",
    )
    args = parser.parse_args()
    run_camera(args.detector, args.checkpoint, args.device, args.api_url)

if __name__ == "__main__":
    main()