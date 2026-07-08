"""Data loading, preprocessing, and augmentation for NEU Surface Defect Database."""

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class NEUDefectDataset(Dataset):
    """PyTorch Dataset for NEU Surface Defect Database.

    Classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
    """

    CLASSES = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
    ):
        """Initialize dataset.

        Args:
            root_dir: Path to NEU dataset root (contains subdirs for each class)
            transform: Optional torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.num_classes = len(self.CLASSES)

        # Load image paths and labels
        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                # Support both .bmp and .jpg formats
                for img_path in (
                    sorted(class_dir.glob("*.bmp"))
                    + sorted(class_dir.glob("*.jpg"))
                    + sorted(class_dir.glob("*.png"))
                ):
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(
    image_size: int = 224,
    augment: bool = False,
) -> transforms.Compose:
    """Get data transforms for training or validation.

    Args:
        image_size: Target image size
        augment: Whether to apply augmentation (for training)

    Returns:
        Composed transforms
    """
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with fixed seed.

    Args:
        data_dir: Path to dataset root
        batch_size: Batch size
        image_size: Target image size
        train_ratio: Proportion for training (0.7 = 70%)
        val_ratio: Proportion for validation (0.15 = 15%)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load full dataset
    full_dataset = NEUDefectDataset(
        root_dir=data_dir,
        transform=get_transforms(image_size=image_size, augment=False),
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    # Apply augmentation to training set
    train_dataset.dataset.transform = get_transforms(image_size=image_size, augment=True)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
