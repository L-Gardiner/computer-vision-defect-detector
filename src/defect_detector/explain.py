"""Grad-CAM visualization for model explainability."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from defect_detector.config import settings
from defect_detector.data import get_transforms
from defect_detector.model import DefectDetectionModel


class GradCAM:
    """Grad-CAM implementation for visualizing model attention."""

    def __init__(self, model: nn.Module, target_layer: str):
        """Initialize Grad-CAM.

        Args:
            model: Model to explain
            target_layer: Name of target layer (e.g., 'backbone.layer4')
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Get target layer
        target = dict(self.model.named_modules())[self.target_layer]
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            x: Input tensor of shape (1, 3, H, W)
            class_idx: Target class index (if None, use predicted class)

        Returns:
            Heatmap of shape (H, W)
        """
        self.model.eval()

        # Forward pass
        with torch.enable_grad():
            x.requires_grad_(True)
            logits = self.model(x)

            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

            # Backward pass
            self.model.zero_grad()
            score = logits[0, class_idx]
            score.backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted sum of activations
        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)  # (H, W)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


def visualize_prediction(
    image_path: str | Path,
    model: DefectDetectionModel,
    device: str = "cpu",
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
) -> tuple[str, float, np.ndarray]:
    """Visualize model prediction with Grad-CAM.

    Args:
        image_path: Path to image
        model: Model to use
        device: Device to use
        class_names: List of class names
        save_path: Path to save visualization

    Returns:
        Tuple of (predicted_class, confidence, heatmap)
    """
    if class_names is None:
        from defect_detector.data import NEUDefectDataset

        class_names = NEUDefectDataset.CLASSES

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(image_size=settings.image_size, augment=False)
    x = transform(image).unsqueeze(0).to(device)

    # Get prediction
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, class_idx = probs.max(dim=1)

    predicted_class = class_names[int(class_idx.item())]  # type: ignore[index]
    confidence = confidence.item()

    # Generate Grad-CAM
    grad_cam = GradCAM(model, target_layer="backbone.layer4")
    heatmap = grad_cam(x, class_idx=int(class_idx.item()))

    # Visualize
    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(heatmap, cmap="hot")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(heatmap, cmap="hot", alpha=0.5)
        axes[2].set_title(
            f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}"
        )
        axes[2].axis("off")

        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return predicted_class, confidence, heatmap
