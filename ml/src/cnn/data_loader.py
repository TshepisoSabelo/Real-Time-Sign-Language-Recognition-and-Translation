from pathlib import Path

import cv2
import numpy as np


def load_dataset(dataset_type="Type_01_(Raw_Gesture)"):
    """Load sign images and one-hot labels from the class-folder dataset.

    Images are resized to ``28x28`` and returned in channel-first format.
    The labels use the class order ``A-Z, DEL, SPACE``.
    """
    dataset_root = Path(__file__).resolve().parents[3] / "dataset" / "Root" / dataset_type
    class_names = [chr(label) for label in range(ord("A"), ord("Z") + 1)] + ["DEL", "SPACE"]
    images = []
    labels = []

    for class_index, class_name in enumerate(class_names):
        class_directory = dataset_root / class_name
        for image_path in sorted(class_directory.iterdir()):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            image = cv2.resize(image, (28, 28)).astype(np.float32) / 255.0
            images.append(np.transpose(image, (2, 0, 1)))

            label = np.zeros(29, dtype=np.float32)
            label[class_index] = 1.0
            labels.append(label)

    if not images:
        raise FileNotFoundError(f"No readable images found in {dataset_root}")

    return np.array(images), np.array(labels)

#x, y = load_dataset(dataset_type="Type_01_(Raw_Gesture)")
