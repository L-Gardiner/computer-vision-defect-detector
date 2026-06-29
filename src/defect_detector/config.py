"""Typed configuration via pydantic-settings.

Single source of truth for runtime settings. Reads from environment
variables and an optional .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App settings
    app_name: str = "defect-detector"
    log_level: str = "INFO"

    # Data paths
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    models_dir: str = "./models"

    # Model architecture
    backbone: str = "resnet18"  # resnet18, mobilenet_v3_small
    num_classes: int = 6  # NEU Surface Defect Database has 6 classes
    image_size: int = 224
    pretrained: bool = True

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 20
    early_stopping_patience: int = 5
    random_seed: int = 42

    # Device
    device: str = "mps"  # mps for macbook, cuda for gpu, cpu for cpu

    @property
    def data_dir_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def models_dir_path(self) -> Path:
        return Path(self.models_dir)


settings = Settings()
