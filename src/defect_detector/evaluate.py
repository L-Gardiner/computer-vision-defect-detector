"""Evaluation metrics and visualization for defect detection model."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from defect_detector.config import settings
from defect_detector.data import NEUDefectDataset
from defect_detector.model import DefectDetectionModel

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    class_names: list[str] | None = None,
) -> dict:
    """Evaluate model on test set.

    Args:
        model: Model to evaluate
        test_loader: Test dataloader
        device: Device to evaluate on
        class_names: List of class names

    Returns:
        Dictionary with metrics
    """
    if class_names is None:
        class_names = NEUDefectDataset.CLASSES

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(all_labels[class_mask], all_preds[class_mask])
            # Use labels parameter to avoid issues with missing classes
            prec, rec, f1_class, _ = precision_recall_fscore_support(
                all_labels[class_mask],
                all_preds[class_mask],
                average=None,
                labels=[i],
                zero_division=0,
            )
            per_class_metrics[class_name] = {
                "accuracy": float(class_acc),
                "precision": float(prec[0]) if isinstance(prec, np.ndarray) else float(prec),  # type: ignore
                "recall": float(rec[0]) if isinstance(rec, np.ndarray) else float(rec),  # type: ignore
                "f1": float(f1_class[0]) if isinstance(f1_class, np.ndarray) else float(f1_class),  # type: ignore
                "support": int(class_mask.sum()),
            }

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1_macro),
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
) -> None:
    """Plot and optionally save confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")

    plt.close()


def main():
    """Main evaluation script."""
    logging.basicConfig(level=logging.INFO)

    # Load data
    from defect_detector.data import get_dataloaders

    _, _, test_loader = get_dataloaders(
        data_dir=settings.raw_data_dir,
        batch_size=settings.batch_size,
        image_size=settings.image_size,
        random_seed=settings.random_seed,
    )

    # Load model
    model_path = Path(settings.models_dir) / "best_model.pt"
    model = DefectDetectionModel.load(
        path=model_path,
        backbone=settings.backbone,
        num_classes=settings.num_classes,
    )
    model = model.to(settings.device)

    # Evaluate
    metrics = evaluate(model, test_loader, device=settings.device)

    # Save metrics
    metrics_path = Path(settings.models_dir) / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Plot confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        class_names=NEUDefectDataset.CLASSES,
        save_path=Path(settings.models_dir) / "confusion_matrix.png",
    )

    # Log metrics
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
    logger.info(f"F1 (macro): {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
