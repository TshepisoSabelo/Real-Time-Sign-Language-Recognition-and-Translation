from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models

CLASS_NAMES = [chr(label) for label in range(ord("A"), ord("Z") + 1)] + ["DEL", "SPACE"]
INPUT_SIZE = 64


class ResNetSignLanguageClassifier:
    """Inference wrapper for the project's 28-class ResNet checkpoint."""

    def __init__(self, checkpoint_path=None, device=None):
        source_root = Path(__file__).resolve().parents[1]
        self.checkpoint_path = Path(
            checkpoint_path or source_root / "models" / "resnet_sign_language.pth"
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_bgr(self, image_bgr, top_k=3):
        """Predicts a BGR OpenCV crop using the training data preprocessing."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Cannot classify an empty image crop")

        image = cv2.resize(
            image_bgr,
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        image = image.astype(np.float32, copy=False) / np.float32(255.0)
        image_chw = np.ascontiguousarray(image.transpose(2, 0, 1)) #changing the image format to CHW (Channel, Height, Width) format
        return self.predict_chw(image_chw, top_k=top_k)

    @torch.inference_mode()
    def predict_chw(self, image_chw, top_k=3):
        """Predict one normalized BGR image in the dataset's CHW format."""
        image_chw = np.asarray(image_chw, dtype=np.float32)
        expected_shape = (3, INPUT_SIZE, INPUT_SIZE)
        if image_chw.shape != expected_shape:
            raise ValueError(f"Expected image shape {expected_shape}, got {image_chw.shape}")

        tensor = torch.from_numpy(np.ascontiguousarray(image_chw)).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        scores, indices = torch.topk(probabilities, k=min(top_k, len(CLASS_NAMES)))
        return [
            (CLASS_NAMES[int(index)], float(score))
            for score, index in zip(scores.cpu(), indices.cpu())
        ]

    @torch.inference_mode()
    def evaluate_dataset(self, dataset, batch_size=32, max_samples=None):
        """Evaluate a lazy SignLanguageDataset and return loss and accuracy."""
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        correct = 0
        samples = 0

        for images, labels, _ in dataset.iter_batches(batch_size=batch_size, shuffle=False):
            if max_samples is not None:
                remaining = max_samples - samples
                if remaining <= 0:
                    break
                images = images[:remaining]
                labels = labels[:remaining]

            inputs = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(self.device)
            targets = torch.from_numpy(np.argmax(labels, axis=1)).long().to(self.device)
            outputs = self.model(inputs)
            total_loss += float(criterion(outputs, targets).item())
            correct += int((outputs.argmax(dim=1) == targets).sum().item())
            samples += len(targets)

            if max_samples is not None and samples >= max_samples:
                break

        if samples == 0:
            raise ValueError("No readable samples were available for evaluation")
        return {
            "samples": samples,
            "loss": total_loss / samples,
            "accuracy": correct / samples,
        }
