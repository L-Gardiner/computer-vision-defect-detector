"""Tests for model architecture and forward pass."""

import tempfile
from pathlib import Path

import pytest
import torch

from defect_detector.model import DefectDetectionModel


@pytest.fixture
def model():
    """Create a test model."""
    return DefectDetectionModel(
        backbone="resnet18",
        num_classes=6,
        pretrained=False,
    )


def test_model_initialization(model):
    """Test model initializes with correct architecture."""
    assert model.num_classes == 6
    assert model.backbone_name == "resnet18"


def test_model_forward_pass(model):
    """Test forward pass returns correct shape."""
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    logits = model(x)

    assert logits.shape == (batch_size, 6)


def test_model_output_dtype(model):  # type: ignore[no-untyped-def]
    """Test model output has correct dtype."""
    x = torch.randn(1, 3, 224, 224)
    logits = model(x)

    assert logits.dtype == torch.float32  # type: ignore[attr-defined]


def test_model_save_and_load():
    """Test model can be saved and loaded."""
    model = DefectDetectionModel(
        backbone="resnet18",
        num_classes=6,
        pretrained=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.pt"
        model.save(save_path)

        assert save_path.exists()

        # Load model
        loaded_model = DefectDetectionModel.load(
            path=save_path,
            backbone="resnet18",
            num_classes=6,
        )

        # Test that loaded model produces same output with same weights
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            original_output = model(x)
            loaded_output = loaded_model(x)

        # Both should have same shape
        assert original_output.shape == loaded_output.shape
        # Weights should be identical after loading
        for p1, p2 in zip(model.parameters(), loaded_model.parameters(), strict=True):
            assert torch.allclose(p1, p2)  # type: ignore[attr-defined]


def test_model_different_backbones():
    """Test model works with different backbones."""
    # Test ResNet18 (primary backbone)
    model = DefectDetectionModel(
        backbone="resnet18",
        num_classes=6,
        pretrained=False,
    )
    x = torch.randn(1, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (1, 6)

    # Test MobileNetV3 (alternative backbone)
    model = DefectDetectionModel(
        backbone="mobilenet_v3_small",
        num_classes=6,
        pretrained=False,
    )
    logits = model(x)
    assert logits.shape == (1, 6)


def test_model_different_num_classes():
    """Test model works with different number of classes."""
    for num_classes in [2, 4, 6, 10]:
        model = DefectDetectionModel(
            backbone="resnet18",
            num_classes=num_classes,
            pretrained=False,
        )
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (1, num_classes)


def test_model_gradient_flow(model):
    """Test that gradients flow through the model."""
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
