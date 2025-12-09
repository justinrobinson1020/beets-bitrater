"""Tests for spectrum analyzer."""


import numpy as np
import pytest

from beetsplug.bitrater.constants import (
    MINIMUM_DURATION,
    MINIMUM_SAMPLE_RATE,
    SPECTRAL_PARAMS,
)
from beetsplug.bitrater.spectrum import SpectrumAnalyzer


class TestSpectrumAnalyzer:
    """Tests for SpectrumAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = SpectrumAnalyzer()

        assert analyzer.num_bands == SPECTRAL_PARAMS["num_bands"]
        assert analyzer.min_freq == SPECTRAL_PARAMS["min_freq"]
        assert analyzer.max_freq == SPECTRAL_PARAMS["max_freq"]
        assert analyzer.fft_size == SPECTRAL_PARAMS["fft_size"]

    def test_band_frequencies(self):
        """Test frequency band calculation."""
        analyzer = SpectrumAnalyzer()

        # Should have 150 bands
        assert len(analyzer._band_frequencies) == 150

        # First band should start at 16 kHz
        assert analyzer._band_frequencies[0][0] == 16000.0

        # Last band should end at ~22.05 kHz
        assert analyzer._band_frequencies[-1][1] == pytest.approx(22050.0, rel=0.01)

        # Band width should be ~40 Hz
        band_width = analyzer._band_frequencies[0][1] - analyzer._band_frequencies[0][0]
        expected_width = (22050 - 16000) / 150
        assert band_width == pytest.approx(expected_width, rel=0.01)

    def test_validate_audio_empty(self):
        """Test validation rejects empty audio (0 samples)."""
        analyzer = SpectrumAnalyzer()
        assert analyzer._validate_audio(np.array([]), MINIMUM_SAMPLE_RATE) is False

    def test_validate_audio_low_sample_rate(self):
        """Test validation rejects sample rate below MINIMUM_SAMPLE_RATE."""
        analyzer = SpectrumAnalyzer()
        y = np.random.rand(44100)  # 1 second of audio
        low_sample_rate = MINIMUM_SAMPLE_RATE - 1  # Just below threshold
        assert analyzer._validate_audio(y, low_sample_rate) is False

    def test_validate_audio_at_minimum_sample_rate(self):
        """Test validation accepts audio at exactly MINIMUM_SAMPLE_RATE."""
        analyzer = SpectrumAnalyzer()
        y = np.random.rand(MINIMUM_SAMPLE_RATE)  # 1 second at minimum rate
        assert analyzer._validate_audio(y, MINIMUM_SAMPLE_RATE) is True

    def test_validate_audio_short_duration(self):
        """Test validation rejects audio below MINIMUM_DURATION threshold."""
        analyzer = SpectrumAnalyzer()
        # Calculate samples just below the minimum duration
        samples_below_threshold = int(MINIMUM_DURATION * MINIMUM_SAMPLE_RATE) - 1
        y = np.random.rand(max(1, samples_below_threshold))
        assert analyzer._validate_audio(y, MINIMUM_SAMPLE_RATE) is False

    def test_validate_audio_at_minimum_duration(self):
        """Test validation accepts audio at exactly MINIMUM_DURATION."""
        analyzer = SpectrumAnalyzer()
        # Calculate samples for exactly the minimum duration
        samples_at_threshold = int(MINIMUM_DURATION * MINIMUM_SAMPLE_RATE) + 1
        y = np.random.rand(samples_at_threshold)
        assert analyzer._validate_audio(y, MINIMUM_SAMPLE_RATE) is True

    def test_validate_audio_valid(self):
        """Test validation accepts valid audio above all thresholds."""
        analyzer = SpectrumAnalyzer()
        y = np.random.rand(MINIMUM_SAMPLE_RATE)  # 1 second at minimum rate
        assert analyzer._validate_audio(y, MINIMUM_SAMPLE_RATE) is True

    def test_extract_band_features_shape(self):
        """Test that extracted features have correct shape."""
        analyzer = SpectrumAnalyzer()

        # Create synthetic PSD data
        freqs = np.linspace(0, 22050, 4097)  # From FFT
        psd = np.random.rand(4097)

        features = analyzer._extract_band_features(psd, freqs)

        assert features is not None
        assert features.shape == (150,)
        assert features.dtype == np.float32

    def test_extract_band_features_normalized(self):
        """Test that features are normalized to 0-1 range."""
        analyzer = SpectrumAnalyzer()

        freqs = np.linspace(0, 22050, 4097)
        psd = np.random.rand(4097) * 1000  # Large values

        features = analyzer._extract_band_features(psd, freqs)

        assert features is not None
        assert np.all(features >= 0)
        assert np.all(features <= 1)
