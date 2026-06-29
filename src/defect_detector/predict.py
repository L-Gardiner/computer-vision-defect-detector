"""Inference utilities for defect detection model."""

from pathlib import Path

import torch
from PIL import Image

from defect_detector.config import settings
from defect_detector.data import NEUDefectDataset, get_transforms
from defect_detector.model import DefectDetectionModel


class Predictor:
    """Wrapper for model inference."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        backbone: str = "resnet18",
        num_classes: int = 6,
    ):
        """Initialize predictor.

        Args:
            model_path: Path to saved model weights
            device: Device to use
            backbone: Backbone architecture
            num_classes: Number of classes
        """
        self.device = device
        self.class_names = NEUDefectDataset.CLASSES
        self.model = DefectDetectionModel.load(
            path=model_path,
            backbone=backbone,
            num_classes=num_classes,
        )
        self.model = self.model.to(device)
        self.model.eval()

    def predict(self, image_path: str | Path) -> dict:
        """Predict class for a single image.

        Args:
            image_path: Path to image

        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        transform = get_transforms(image_size=settings.image_size, augment=False)
        x = transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

        # Get top predictions
        top_probs, top_indices = torch.topk(probs[0], k=3)

        results = {
            "predicted_class": self.class_names[int(top_indices[0].item())],  # type: ignore[index]
            "confidence": top_probs[0].item(),
            "top_3": [
                {
                    "class": self.class_names[int(idx.item())],
                    "confidence": prob.item(),
                }
                for prob, idx in zip(top_probs, top_indices, strict=True)
            ],
        }

        return results
