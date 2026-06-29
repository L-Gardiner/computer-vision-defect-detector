"""Tests for prediction and inference."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from defect_detector.model import DefectDetectionModel
from defect_detector.predict import Predictor


@pytest.fixture
def dummy_model():
    """Create a dummy trained model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model = DefectDetectionModel(
            backbone="resnet18",
            num_classes=6,
            pretrained=False,
        )
        model_path = tmpdir / "model.pt"
        model.save(model_path)
        yield model_path


@pytest.fixture
def dummy_image():
    """Create a dummy test image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img = Image.new("RGB", (224, 224), color="red")
        img_path = tmpdir / "test_image.jpg"
        img.save(img_path)
        yield img_path


def test_predictor_initialization(dummy_model):
    """Test predictor initializes correctly."""
    predictor = Predictor(
        model_path=dummy_model,
        device="cpu",
        backbone="resnet18",
        num_classes=6,
    )

    assert predictor.device == "cpu"
    assert len(predictor.class_names) == 6


def test_predictor_predict(dummy_model, dummy_image):
    """Test predictor makes predictions."""
    predictor = Predictor(
        model_path=dummy_model,
        device="cpu",
        backbone="resnet18",
        num_classes=6,
    )

    results = predictor.predict(dummy_image)

    assert "predicted_class" in results
    assert "confidence" in results
    assert "top_3" in results

    assert isinstance(results["predicted_class"], str)
    assert isinstance(results["confidence"], float)
    assert 0.0 <= results["confidence"] <= 1.0

    assert len(results["top_3"]) == 3
    for pred in results["top_3"]:
        assert "class" in pred
        assert "confidence" in pred


def test_prediction_confidence_valid(dummy_model, dummy_image):
    """Test that prediction confidence is valid."""
    predictor = Predictor(
        model_path=dummy_model,
        device="cpu",
        backbone="resnet18",
        num_classes=6,
    )

    results = predictor.predict(dummy_image)

    # Confidence should be between 0 and 1
    assert 0.0 <= results["confidence"] <= 1.0

    # Top 3 should be in descending order
    confidences = [p["confidence"] for p in results["top_3"]]
    assert confidences == sorted(confidences, reverse=True)


def test_prediction_class_valid(dummy_model, dummy_image):
    """Test that predicted class is valid."""
    predictor = Predictor(
        model_path=dummy_model,
        device="cpu",
        backbone="resnet18",
        num_classes=6,
    )

    results = predictor.predict(dummy_image)

    assert results["predicted_class"] in predictor.class_names

    for pred in results["top_3"]:
        assert pred["class"] in predictor.class_names
