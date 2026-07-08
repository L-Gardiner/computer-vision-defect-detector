"""Transfer learning model architecture for defect detection."""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


class DefectDetectionModel(nn.Module):
    """Transfer learning model with pretrained backbone and custom head."""

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 6,
        pretrained: bool = True,
    ):
        """Initialize model.

        Args:
            backbone: Backbone architecture (resnet18, mobilenet_v3_small)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # Load pretrained backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original head
        elif backbone == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
            # MobileNetV3 has a sequential classifier
            # Get input features from the first layer of the classifier
            in_features = self.backbone.classifier[0].in_features
            # Replace classifier with identity
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def save(self, path: str | Path) -> None:
        """Save model weights.

        Args:
            path: Path to save weights
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        backbone: str = "resnet18",
        num_classes: int = 6,
    ) -> "DefectDetectionModel":
        """Load model from weights.

        Args:
            path: Path to saved weights
            backbone: Backbone architecture
            num_classes: Number of classes

        Returns:
            Loaded model
        """
        model = cls(backbone=backbone, num_classes=num_classes, pretrained=False)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model
