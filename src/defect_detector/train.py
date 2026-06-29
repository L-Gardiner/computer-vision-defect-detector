"""Training loop for defect detection model."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from defect_detector.config import settings
from defect_detector.model import DefectDetectionModel

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Validate model.

    Args:
        model: Model to validate
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)  # type: ignore[attr-defined]
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    model: DefectDetectionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 5,
    device: str = "cpu",
    save_dir: str | Path = "./models",
) -> dict:
    """Train model with early stopping.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        early_stopping_patience: Patience for early stopping
        device: Device to train on
        save_dir: Directory to save best model

    Returns:
        Dictionary with training history
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)

        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(save_dir / "best_model.pt")
            logger.info(f"Saved best model with val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(patience={early_stopping_patience})"
                )
                break

    return history


def main():
    """Main training script."""
    logging.basicConfig(level=logging.INFO)

    # Load data
    from defect_detector.data import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=settings.raw_data_dir,
        batch_size=settings.batch_size,
        image_size=settings.image_size,
        random_seed=settings.random_seed,
    )

    # Create model
    model = DefectDetectionModel(
        backbone=settings.backbone,
        num_classes=settings.num_classes,
        pretrained=settings.pretrained,
    )

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=settings.num_epochs,
        learning_rate=settings.learning_rate,
        early_stopping_patience=settings.early_stopping_patience,
        device=settings.device,
        save_dir=settings.models_dir,
    )

    # Save history
    history_path = Path(settings.models_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
