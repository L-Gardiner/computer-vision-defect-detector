"""Streamlit app for defect detection demo."""

from pathlib import Path

import streamlit as st
from PIL import Image

from defect_detector.config import settings
from defect_detector.explain import visualize_prediction
from defect_detector.model import DefectDetectionModel
from defect_detector.predict import Predictor


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Defect Detector",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Defect Detection with Grad-CAM")
    st.markdown(
        """
    Upload an image to classify surface defects using a transfer learning model
    with explainability via Grad-CAM heatmaps.
    """
    )

    # Sidebar
    st.sidebar.header("Settings")
    model_path = Path(settings.models_dir) / "best_model.pt"

    if not model_path.exists():
        st.error(
            f"❌ Model not found at {model_path}. "
            "Please train the model first using `python -m defect_detector.train`"
        )
        return

    # Load model
    try:
        predictor = Predictor(
            model_path=model_path,
            device=settings.device,
            backbone=settings.backbone,
            num_classes=settings.num_classes,
        )
        model = DefectDetectionModel.load(
            path=model_path,
            backbone=settings.backbone,
            num_classes=settings.num_classes,
        )
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return

    # File upload
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        # Make prediction
        with st.spinner("🔄 Analyzing image..."):
            # Save temp image
            temp_path = Path("/tmp/temp_image.jpg")
            image.save(temp_path)

            # Get prediction
            results = predictor.predict(temp_path)

            # Generate Grad-CAM visualization
            viz_path = Path("/tmp/grad_cam_viz.png")
            predicted_class, confidence, heatmap = visualize_prediction(
                image_path=temp_path,
                model=model,
                device=settings.device,
                save_path=viz_path,
            )

        with col2:
            st.subheader("🎯 Prediction Results")

            # Main prediction
            st.metric(
                "Predicted Class",
                results["predicted_class"],
                f"{results['confidence']:.1%}",
            )

            # Top 3 predictions
            st.subheader("Top 3 Predictions")
            for i, pred in enumerate(results["top_3"], 1):
                st.write(
                    f"{i}. **{pred['class']}**: {pred['confidence']:.1%}"
                )

        # Grad-CAM visualization
        st.subheader("🔥 Grad-CAM Explainability")
        st.image(viz_path, use_column_width=True)
        st.markdown(
            """
            The Grad-CAM heatmap shows where the model focuses when making its prediction.
            Brighter regions indicate areas the model considers most important for classification.
            """
        )

        # Cleanup
        temp_path.unlink(missing_ok=True)
        viz_path.unlink(missing_ok=True)

    # Info section
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown(
        """
        **Model Details:**
        - Backbone: ResNet18 (pretrained)
        - Classes: 6 defect types
        - Explainability: Grad-CAM
        
        **Classes:**
        - Crazing
        - Inclusion
        - Patches
        - Pitted Surface
        - Rolled-in Scale
        - Scratches
        """
    )


if __name__ == "__main__":
    main()
