"""Tests for data loading and preprocessing."""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from defect_detector.data import NEUDefectDataset, get_transforms


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy images for each class
        for class_name in NEUDefectDataset.CLASSES:
            class_dir = tmpdir / class_name
            class_dir.mkdir()

            # Create 3 dummy images per class
            for i in range(3):
                img = Image.new("RGB", (200, 200), color="red")
                img.save(class_dir / f"image_{i}.bmp")

        yield tmpdir


def test_dataset_initialization(dummy_dataset):
    """Test dataset initializes correctly."""
    dataset = NEUDefectDataset(root_dir=dummy_dataset)

    assert len(dataset) == 18  # 6 classes * 3 images
    assert dataset.num_classes == 6


def test_dataset_getitem(dummy_dataset):
    """Test dataset returns correct item format."""
    dataset = NEUDefectDataset(root_dir=dummy_dataset)
    image, label = dataset[0]

    assert isinstance(image, Image.Image)
    assert isinstance(label, int)
    assert 0 <= label < 6


def test_dataset_with_transforms(dummy_dataset):
    """Test dataset with transforms returns tensor."""
    transform = get_transforms(image_size=224, augment=False)
    dataset = NEUDefectDataset(root_dir=dummy_dataset, transform=transform)

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32  # type: ignore[attr-defined]


def test_transforms_output_shape():
    """Test transforms produce correct output shape."""
    transform = get_transforms(image_size=224, augment=False)

    # Create dummy image
    img = Image.new("RGB", (300, 300), color="blue")

    # Apply transforms
    tensor: torch.Tensor = transform(img)  # type: ignore[assignment]

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32  # type: ignore[attr-defined]
    assert tensor.min() >= -3.0  # Normalized (ImageNet stats)
    assert tensor.max() <= 3.0


def test_augmentation_transforms():
    """Test augmentation transforms work."""
    transform = get_transforms(image_size=224, augment=True)

    img = Image.new("RGB", (300, 300), color="green")
    tensor: torch.Tensor = transform(img)  # type: ignore[assignment]

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32  # type: ignore[attr-defined]


def test_transforms_normalization():
    """Test that transforms normalize correctly."""
    transform = get_transforms(image_size=224, augment=False)

    # Create a white image (should be close to 1 after normalization)
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    tensor: torch.Tensor = transform(img)  # type: ignore[assignment]

    # After normalization with ImageNet stats, white should be around 2.6
    assert tensor.mean() > 2.0


def test_classes_list():
    """Test that class names are correct."""
    expected_classes = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]
    assert expected_classes == NEUDefectDataset.CLASSES
