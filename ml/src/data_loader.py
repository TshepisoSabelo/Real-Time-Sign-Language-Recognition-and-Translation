from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

CLASS_NAMES = [chr(label) for label in range(ord("A"), ord("Z") + 1)] + ["DEL", "SPACE"]


class SignLanguageDataset:
    """Lazy image dataset that stores paths and loads one mini-batch at a time."""

    def __init__(self, dataset_type="Type_01_(Raw_Gesture)", size=64, transpose=False):
        self.size = size
        self.transpose = transpose
        self.image_paths, self.labels = self._index_dataset(dataset_type)

    def _index_dataset(self, dataset_type):
        """Index files without decoding images or allocating image arrays."""
        dataset_root = Path(__file__).resolve().parents[2] / "dataset" / "Root" / dataset_type
        image_paths = []
        labels = []

        for class_index, class_name in enumerate(CLASS_NAMES):
            class_directory = dataset_root / class_name
            if not class_directory.is_dir():
                continue
            for image_path in sorted(class_directory.iterdir()):
                if image_path.is_file():
                    image_paths.append(image_path)
                    labels.append(class_index)

        if not image_paths:
            raise FileNotFoundError(f"No image files found in {dataset_root}")
        return np.asarray(image_paths, dtype=object), np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.image_paths)

    def subset(self, indices):
        subset = object.__new__(SignLanguageDataset)
        subset.size = self.size
        subset.transpose = self.transpose
        subset.image_paths = self.image_paths[indices]
        subset.labels = self.labels[indices]
        return subset

    def _load_image(self, image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32, copy=False) / np.float32(255.0)
        if self.transpose:
            image = image.transpose(2, 0, 1)
        return np.ascontiguousarray(image, dtype=np.float32)

    def load_batch(self, indices):
        """Decode a batch, skipping unreadable files without aborting training."""
        images = []
        labels = []
        load_start = perf_counter()
        for index in indices:
            image = self._load_image(self.image_paths[index])
            if image is None:
                continue
            images.append(image)
            labels.append(self.labels[index])

        if not images:
            return None, None, perf_counter() - load_start

        batch_images = np.stack(images).astype(np.float32, copy=False)
        batch_labels = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float32)
        batch_labels[np.arange(len(labels)), labels] = np.float32(1.0)
        return batch_images, batch_labels, perf_counter() - load_start

    def iter_batches(self, batch_size=32, shuffle=True, drop_last=False):
        """Yield ``(images, labels, loading_seconds)`` mini-batches."""
        indices = np.arange(len(self), dtype=np.int64)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            if drop_last and len(batch_indices) < batch_size:
                continue
            batch_images, batch_labels, load_seconds = self.load_batch(batch_indices)
            if batch_images is not None:
                yield batch_images, batch_labels, load_seconds


def build_dataset(dataset_type="Type_01_(Raw_Gesture)", size=64, transpose=False):
    """Build a lazy dataset index without decoding the complete dataset."""
    return SignLanguageDataset(dataset_type=dataset_type, size=size, transpose=transpose)


def load_dataset(dataset_type="Type_01_(Raw_Gesture)", size=64, transpose=False):
    """Compatibility helper that eagerly materializes the lazy dataset."""
    dataset = build_dataset(dataset_type=dataset_type, size=size, transpose=transpose)
    dataset_root = Path(__file__).resolve().parents[2] / "dataset" / "Root" / dataset_type
    images = []
    labels = []
    for batch_images, batch_labels, _ in dataset.iter_batches(batch_size=256, shuffle=False):
        images.append(batch_images)
        labels.append(batch_labels)
    return np.concatenate(images).astype(np.float32, copy=False), np.concatenate(labels).astype(np.float32, copy=False)

#x, y = load_dataset(dataset_type="Type_01_(Raw_Gesture)")

#print(x.shape)
