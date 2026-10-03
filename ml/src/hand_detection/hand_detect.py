import pickle as plk

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import pathlib
import argparse
from pathlib import Path
from resnet_inference import ResNetSignLanguageClassifier
#==============================================================================
#Open Camera object
#xmodel = YOLO("yolo26n.pt")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_PATH = PROJECT_ROOT / "ml" / "src" / "models"
DEFAULT_DETECTOR = PROJECT_ROOT / "runs" / "pose" / "train-9" / "weights" / "best.pt"
DEFAULT_CHECKPOINT = DEFAULT_MODELS_PATH / "resnet_sign_language.pth"

def run_camera(detector_path=DEFAULT_DETECTOR, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
    """Runs hand detection followed by ResNet sign classification."""
    detection_model = YOLO(str(detector_path))
    classifier = ResNetSignLanguageClassifier(checkpoint_path=checkpoint_path, device=device)
    with open(DEFAULT_MODELS_PATH / "comparator_model", "rb") as f:
        comparator_model = plk.load(f)

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            success, image = capture.read()
            if not success:
                print("Unable to read a frame from the camera")
                break

            image = cv2.flip(image, 1)
            result = detection_model(image, verbose=False)[0]
            letter = "None"
            best_confidence = 0.0

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
                #predictions = classifier.predict_bgr(hand, top_k=1)
                #predictions = comparator_model.predict(hand, top_k=1)
                #letter, class_confidence = predictions[0]
                letter = "A" # Placeholder for actual prediction
                best_confidence = confidence
                cv2.imshow("Hand", hand)
                cv2.putText(
                    image,
                    f"{letter} {best_confidence:.2f}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            annotated = result.plot()
            cv2.putText(
                annotated,
                str(letter),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Hand Tracker", annotated)

            if cv2.waitKey(1) & 0xFF == ord("x"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Run hand detection and ResNet sign prediction")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda")
    args = parser.parse_args()
    run_camera(args.detector, args.checkpoint, args.device)


if __name__ == "__main__":
    main()


