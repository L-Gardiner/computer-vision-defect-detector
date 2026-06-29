"""Tests for FastAPI endpoints."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from defect_detector.api import app
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
        model_path = tmpdir / "best_model.pt"
        model.save(model_path)

        # Patch settings to use our temp model
        import defect_detector.api as api_module

        original_model_path = api_module.model_path
        api_module.model_path = model_path

        yield model_path

        # Restore
        api_module.model_path = original_model_path


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_no_file(client):
    """Test predict endpoint without file."""
    response = client.post("/predict")

    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_with_image(client, dummy_model):
    """Test predict endpoint with image file."""
    # Load predictor manually for testing
    import defect_detector.api as api_module

    api_module.predictor = Predictor(
        model_path=dummy_model,
        device="cpu",
        backbone="resnet18",
        num_classes=6,
    )

    # Create dummy image
    img = Image.new("RGB", (224, 224), color="red")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )

        assert response.status_code == 200
        data = response.json()

        assert "predicted_class" in data
        assert "confidence" in data
        assert "top_3" in data

        assert isinstance(data["predicted_class"], str)
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    finally:
        Path(tmp_path).unlink()


def test_api_response_schema(client, dummy_model):
    """Test API response follows expected schema."""
    # Load predictor manually for testing
    import defect_detector.api as api_module

    api_module.predictor = Predictor(
        model_path=dummy_model,
        device="cpu",
        backbone="resnet18",
        num_classes=6,
    )

    img = Image.new("RGB", (224, 224), color="blue")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )

        data = response.json()

        # Check top_3 structure
        assert len(data["top_3"]) == 3
        for pred in data["top_3"]:
            assert "class" in pred
            assert "confidence" in pred
            assert isinstance(pred["class"], str)
            assert isinstance(pred["confidence"], float)

    finally:
        Path(tmp_path).unlink()
