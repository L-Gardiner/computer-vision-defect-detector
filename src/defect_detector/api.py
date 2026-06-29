"""FastAPI inference service for defect detection."""

from pathlib import Path

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from defect_detector import __version__
from defect_detector.config import settings
from defect_detector.predict import Predictor

app = FastAPI(
    title="Defect Detector API",
    version=__version__,
    description="Transfer learning image classification API with Grad-CAM explainability",
)

# Load model at startup
model_path = Path(settings.models_dir) / "best_model.pt"
predictor = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global predictor
    if model_path.exists():
        predictor = Predictor(
            model_path=model_path,
            device=settings.device,
            backbone=settings.backbone,
            num_classes=settings.num_classes,
        )


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    predicted_class: str
    confidence: float
    top_3: list[dict[str, str | float]]


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile) -> PredictionResponse:
    """Predict defect class from uploaded image.

    Args:
        file: Uploaded image file

    Returns:
        Prediction results with confidence and top-3 predictions
    """
    if predictor is None:
        return {
            "predicted_class": "error",
            "confidence": 0.0,
            "top_3": [],
        }

    # Read uploaded file
    contents = await file.read()
    temp_path = Path("/tmp") / file.filename

    # Save temporarily
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        # Make prediction
        results = predictor.predict(temp_path)
        return PredictionResponse(**results)
    finally:
        # Cleanup
        temp_path.unlink(missing_ok=True)
