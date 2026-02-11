"""Pytest configuration and fixtures for bitrater tests."""

from pathlib import Path

import numpy as np
import pytest

from bitrater.constants import SPECTRAL_PARAMS
from bitrater.types import SpectralFeatures


@pytest.fixture
def sample_features() -> SpectralFeatures:
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
        (min_freq + i * band_width, min_freq + (i + 1) * band_width) for i in range(num_bands)
    ]

    # Realistic 320kbps SFB21 values: moderate ultra ratio, good continuity
    sfb21_features = np.array([0.15, 0.6, 0.3, 0.05, 0.08, 0.25], dtype=np.float32)
    # Realistic 320kbps rolloff: moderate slope, some drop
    rolloff_features = np.array([-0.8, -12.0, 0.7, 0.3], dtype=np.float32)
    # Realistic 320kbps discriminative: moderate energy, balanced ratios
    discriminative_features = np.array(
        [1.5, 2.0, 25.0, 5.0, 0.15, 2.5], dtype=np.float32
    )

    return SpectralFeatures(
        features=features,
        frequency_bands=frequency_bands,
        cutoff_features=np.zeros(6, dtype=np.float32),
        sfb21_features=sfb21_features,
        rolloff_features=rolloff_features,
        discriminative_features=discriminative_features,
        is_vbr=0.0,  # Simulating CBR file
    )


@pytest.fixture
def lossless_features() -> SpectralFeatures:
    """Create SpectralFeatures simulating lossless audio."""
    num_bands = SPECTRAL_PARAMS["num_bands"]

    # Lossless has content across all bands including ultrasonic
    features = np.linspace(0.9, 0.4, num_bands).astype(np.float32)

    min_freq = SPECTRAL_PARAMS["min_freq"]
    max_freq = SPECTRAL_PARAMS["max_freq"]
    band_width = (max_freq - min_freq) / num_bands

    frequency_bands = [
        (min_freq + i * band_width, min_freq + (i + 1) * band_width) for i in range(num_bands)
    ]

    # Realistic lossless SFB21 values: high ultra ratio, strong continuity
    sfb21_features = np.array([0.6, 0.85, 0.7, 0.02, 0.03, 0.65], dtype=np.float32)
    # Realistic lossless rolloff: shallow slope, minimal drop
    rolloff_features = np.array([-0.2, -3.0, 0.9, 0.8], dtype=np.float32)
    # Realistic lossless discriminative: high energy, balanced ratios
    discriminative_features = np.array(
        [1.2, 1.3, 45.0, 18.0, 0.45, 1.8], dtype=np.float32
    )

    return SpectralFeatures(
        features=features,
        frequency_bands=frequency_bands,
        cutoff_features=np.zeros(6, dtype=np.float32),
        sfb21_features=sfb21_features,
        rolloff_features=rolloff_features,
        discriminative_features=discriminative_features,
        is_vbr=0.0,  # Lossless is not VBR
    )


@pytest.fixture
def temp_model_path(tmp_path: Path) -> Path:
    """Create a temporary path for saving/loading models."""
    return tmp_path / "test_model.pkl"


@pytest.fixture
def training_features() -> tuple[list, list]:
    """Create a set of training features for different classes.

    Classes (by index):
    - 0: 128 kbps - sharp cutoff early
    - 1: V2 - cutoff at ~18.5 kHz
    - 2: 192 kbps - moderate cutoff
    - 3: V0 - high bitrate VBR
    - 4: 256 kbps - good quality
    - 5: 320 kbps - best CBR quality
    - 6: Lossless - full spectrum
    """
    from enum import IntEnum

    class TrainingClass(IntEnum):
        """Class indices for training data."""

        CBR_128 = 0
        VBR_V2 = 1
        CBR_192 = 2
        VBR_V0 = 3
        CBR_256 = 4
        CBR_320 = 5
        LOSSLESS = 6

    rng = np.random.default_rng(42)
    num_bands = SPECTRAL_PARAMS["num_bands"]
    features_list = []
    labels = []

    # Create 10 samples per class (7 classes)
    for class_idx in TrainingClass:
        for _ in range(10):
            features = rng.random(num_bands).astype(np.float32)

            # Add class-specific characteristics based on bitrate cutoff frequencies
            if class_idx == TrainingClass.CBR_128:  # 128 kbps - sharp cutoff early
                features[60:] *= 0.1
            elif class_idx == TrainingClass.VBR_V2:  # V2 - cutoff at ~18.5 kHz
                features[65:] *= 0.12
            elif class_idx == TrainingClass.CBR_192:  # 192 kbps
                features[70:] *= 0.15
            elif class_idx == TrainingClass.VBR_V0:  # V0
                features[75:] *= 0.18
            elif class_idx == TrainingClass.CBR_256:  # 256 kbps
                features[80:] *= 0.2
            elif class_idx == TrainingClass.CBR_320:  # 320 kbps
                features[90:] *= 0.25
            elif class_idx == TrainingClass.LOSSLESS:  # Lossless - content in ultrasonic
                features[100:] *= 0.8

            min_freq = SPECTRAL_PARAMS["min_freq"]
            max_freq = SPECTRAL_PARAMS["max_freq"]
            band_width = (max_freq - min_freq) / num_bands

            frequency_bands = [
                (min_freq + i * band_width, min_freq + (i + 1) * band_width)
                for i in range(num_bands)
            ]

            # V2 and V0 are VBR, others are CBR
            is_vbr = 1.0 if class_idx in [TrainingClass.VBR_V2, TrainingClass.VBR_V0] else 0.0

            # Class-appropriate SFB21 features (ultra_ratio, continuity, flatness, flat_std, flat_iqr, flat_19_20k)
            noise = rng.random(6).astype(np.float32) * 0.05
            if class_idx == TrainingClass.CBR_128:
                sfb21 = np.array([0.02, 0.2, 0.1, 0.01, 0.02, 0.05], dtype=np.float32) + noise
            elif class_idx == TrainingClass.VBR_V2:
                sfb21 = np.array([0.05, 0.35, 0.15, 0.03, 0.04, 0.10], dtype=np.float32) + noise
            elif class_idx == TrainingClass.CBR_192:
                sfb21 = np.array([0.08, 0.45, 0.20, 0.02, 0.03, 0.15], dtype=np.float32) + noise
            elif class_idx == TrainingClass.VBR_V0:
                sfb21 = np.array([0.10, 0.55, 0.25, 0.08, 0.10, 0.20], dtype=np.float32) + noise
            elif class_idx == TrainingClass.CBR_256:
                sfb21 = np.array([0.12, 0.58, 0.28, 0.04, 0.06, 0.22], dtype=np.float32) + noise
            elif class_idx == TrainingClass.CBR_320:
                sfb21 = np.array([0.15, 0.60, 0.30, 0.05, 0.08, 0.25], dtype=np.float32) + noise
            else:  # LOSSLESS
                sfb21 = np.array([0.60, 0.85, 0.70, 0.02, 0.03, 0.65], dtype=np.float32) + noise

            # Class-appropriate rolloff features (slope, total_drop, ratio_early, ratio_late)
            noise_r = rng.random(4).astype(np.float32) * 0.1
            if class_idx == TrainingClass.CBR_128:
                rolloff = np.array([-2.5, -30.0, 0.2, 0.05], dtype=np.float32) + noise_r
            elif class_idx == TrainingClass.VBR_V2:
                rolloff = np.array([-1.8, -22.0, 0.35, 0.10], dtype=np.float32) + noise_r
            elif class_idx == TrainingClass.CBR_192:
                rolloff = np.array([-1.5, -18.0, 0.45, 0.15], dtype=np.float32) + noise_r
            elif class_idx == TrainingClass.VBR_V0:
                rolloff = np.array([-1.2, -15.0, 0.55, 0.20], dtype=np.float32) + noise_r
            elif class_idx == TrainingClass.CBR_256:
                rolloff = np.array([-1.0, -13.0, 0.65, 0.25], dtype=np.float32) + noise_r
            elif class_idx == TrainingClass.CBR_320:
                rolloff = np.array([-0.8, -12.0, 0.70, 0.30], dtype=np.float32) + noise_r
            else:  # LOSSLESS
                rolloff = np.array([-0.2, -3.0, 0.90, 0.80], dtype=np.float32) + noise_r

            # Class-appropriate discriminative features
            noise_d = rng.random(6).astype(np.float32) * 0.1
            if class_idx == TrainingClass.CBR_128:
                discrim = np.array([8.0, 10.0, 5.0, 0.5, 0.02, 3.0], dtype=np.float32) + noise_d
            elif class_idx == TrainingClass.VBR_V2:
                discrim = np.array([3.0, 4.0, 18.0, 3.0, 0.08, 2.8], dtype=np.float32) + noise_d
            elif class_idx == TrainingClass.CBR_192:
                discrim = np.array([2.5, 3.0, 22.0, 4.0, 0.10, 2.5], dtype=np.float32) + noise_d
            elif class_idx == TrainingClass.VBR_V0:
                discrim = np.array([2.0, 2.5, 28.0, 6.0, 0.12, 2.3], dtype=np.float32) + noise_d
            elif class_idx == TrainingClass.CBR_256:
                discrim = np.array([1.8, 2.2, 32.0, 8.0, 0.18, 2.2], dtype=np.float32) + noise_d
            elif class_idx == TrainingClass.CBR_320:
                discrim = np.array([1.5, 2.0, 35.0, 10.0, 0.20, 2.5], dtype=np.float32) + noise_d
            else:  # LOSSLESS
                discrim = np.array([1.2, 1.3, 45.0, 18.0, 0.45, 1.8], dtype=np.float32) + noise_d

            features_list.append(
                SpectralFeatures(
                    features=features,
                    frequency_bands=frequency_bands,
                    cutoff_features=np.zeros(6, dtype=np.float32),
                    sfb21_features=sfb21,
                    rolloff_features=rolloff,
                    discriminative_features=discrim,
                    is_vbr=is_vbr,
                )
            )
            labels.append(int(class_idx))

    return features_list, labels
