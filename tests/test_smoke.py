"""Smoke tests: the package imports and core wiring exists."""

import importlib


def test_package_imports() -> None:
    """Test that package imports successfully."""
    mod = importlib.import_module("defect_detector")
    assert hasattr(mod, "__version__")


def test_config_loads() -> None:
    """Test that config loads with expected settings."""
    from defect_detector.config import settings

    assert settings.app_name == "defect-detector"
    assert settings.num_classes == 6
    assert settings.image_size == 224
