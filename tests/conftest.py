"""Pytest configuration and fixtures for bitrater tests."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from beetsplug.bitrater.types import SpectralFeatures, ClassifierPrediction
from beetsplug.bitrater.constants import SPECTRAL_PARAMS


@pytest.fixture
def sample_features():
    """Create sample SpectralFeatures for testing."""
    num_bands = SPECTRAL_PARAMS["num_bands"]

    # Create realistic-looking PSD features
    # Simulate a lossy file with cutoff around band 100 (20 kHz)
    features = np.zeros(num_bands, dtype=np.float32)

    # Lower bands (16-20 kHz) have content
    features[:100] = np.linspace(0.8, 0.3, 100)

    # Upper bands (20-22 kHz) have little content (simulating lossy)
    features[100:] = np.linspace(0.1, 0.01, 50)

    # Calculate band frequencies
    min_freq = SPECTRAL_PARAMS["min_freq"]
    max_freq = SPECTRAL_PARAMS["max_freq"]
    band_width = (max_freq - min_freq) / num_bands

    frequency_bands = [
        (min_freq + i * band_width, min_freq + (i + 1) * band_width)
        for i in range(num_bands)
    ]

    return SpectralFeatures(
        features=features,
        frequency_bands=frequency_bands
    )


@pytest.fixture
def lossless_features():
    """Create SpectralFeatures simulating lossless audio."""
    num_bands = SPECTRAL_PARAMS["num_bands"]

    # Lossless has content across all bands including ultrasonic
    features = np.linspace(0.9, 0.4, num_bands).astype(np.float32)

    min_freq = SPECTRAL_PARAMS["min_freq"]
    max_freq = SPECTRAL_PARAMS["max_freq"]
    band_width = (max_freq - min_freq) / num_bands

    frequency_bands = [
        (min_freq + i * band_width, min_freq + (i + 1) * band_width)
        for i in range(num_bands)
    ]

    return SpectralFeatures(
        features=features,
        frequency_bands=frequency_bands
    )


@pytest.fixture
def temp_model_path(tmp_path):
    """Create a temporary path for saving/loading models."""
    return tmp_path / "test_model.pkl"


@pytest.fixture
def training_features():
    """Create a set of training features for different classes."""
    num_bands = SPECTRAL_PARAMS["num_bands"]
    features_list = []
    labels = []

    # Create 10 samples per class (7 classes)
    for class_idx in range(7):
        for _ in range(10):
            features = np.random.rand(num_bands).astype(np.float32)

            # Add class-specific characteristics
            if class_idx == 0:  # 128 kbps - sharp cutoff early
                features[60:] *= 0.1
            elif class_idx == 1:  # V2 - cutoff at ~18.5 kHz
                features[65:] *= 0.12
            elif class_idx == 2:  # 192 kbps
                features[70:] *= 0.15
            elif class_idx == 3:  # V0
                features[75:] *= 0.18
            elif class_idx == 4:  # 256 kbps
                features[80:] *= 0.2
            elif class_idx == 5:  # 320 kbps
                features[90:] *= 0.25
            else:  # Lossless - content in ultrasonic
                features[100:] *= 0.8

            min_freq = SPECTRAL_PARAMS["min_freq"]
            max_freq = SPECTRAL_PARAMS["max_freq"]
            band_width = (max_freq - min_freq) / num_bands

            frequency_bands = [
                (min_freq + i * band_width, min_freq + (i + 1) * band_width)
                for i in range(num_bands)
            ]

            features_list.append(SpectralFeatures(
                features=features,
                frequency_bands=frequency_bands
            ))
            labels.append(class_idx)

    return features_list, labels
