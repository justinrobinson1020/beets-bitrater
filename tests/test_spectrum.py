"""Tests for spectrum analyzer."""


import numpy as np
import pytest

from beetsplug.bitrater.constants import (
    MINIMUM_DURATION,
    MINIMUM_SAMPLE_RATE,
    SPECTRAL_PARAMS,
)
from beetsplug.bitrater.spectrum import SpectrumAnalyzer
from beetsplug.bitrater.types import SpectralFeatures


class TestSpectrumAnalyzer:
    """Tests for SpectrumAnalyzer class."""

    def test_init(self) -> None:
        """Test analyzer initialization."""
        analyzer = SpectrumAnalyzer()

        assert analyzer.num_bands == SPECTRAL_PARAMS["num_bands"]
        assert analyzer.min_freq == SPECTRAL_PARAMS["min_freq"]
        assert analyzer.max_freq == SPECTRAL_PARAMS["max_freq"]
        assert analyzer.fft_size == SPECTRAL_PARAMS["fft_size"]

    def test_band_frequencies(self) -> None:
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

    def test_validate_audio_empty(self) -> None:
        """Test validation rejects empty audio (0 samples)."""
        analyzer = SpectrumAnalyzer()
        assert analyzer._validate_audio(np.array([]), MINIMUM_SAMPLE_RATE) is False

    def test_validate_audio_low_sample_rate(self) -> None:
        """Test validation rejects sample rate below MINIMUM_SAMPLE_RATE."""
        analyzer = SpectrumAnalyzer()
        y = np.random.rand(44100)  # 1 second of audio
        low_sample_rate = MINIMUM_SAMPLE_RATE - 1  # Just below threshold
        assert analyzer._validate_audio(y, low_sample_rate) is False

    def test_validate_audio_at_minimum_sample_rate(self) -> None:
        """Test validation accepts audio at exactly MINIMUM_SAMPLE_RATE."""
        analyzer = SpectrumAnalyzer()
        y = np.random.rand(MINIMUM_SAMPLE_RATE)  # 1 second at minimum rate
        assert analyzer._validate_audio(y, MINIMUM_SAMPLE_RATE) is True

    def test_validate_audio_short_duration(self) -> None:
        """Test validation rejects audio below MINIMUM_DURATION threshold."""
        analyzer = SpectrumAnalyzer()
        # Calculate samples just below the minimum duration
        samples_below_threshold = int(MINIMUM_DURATION * MINIMUM_SAMPLE_RATE) - 1
        y = np.random.rand(max(1, samples_below_threshold))
        assert analyzer._validate_audio(y, MINIMUM_SAMPLE_RATE) is False

    def test_validate_audio_at_minimum_duration(self) -> None:
        """Test validation accepts audio at exactly MINIMUM_DURATION."""
        analyzer = SpectrumAnalyzer()
        # Calculate samples for exactly the minimum duration
        samples_at_threshold = int(MINIMUM_DURATION * MINIMUM_SAMPLE_RATE) + 1
        y = np.random.rand(samples_at_threshold)
        assert analyzer._validate_audio(y, MINIMUM_SAMPLE_RATE) is True

    def test_validate_audio_valid(self) -> None:
        """Test validation accepts valid audio above all thresholds."""
        analyzer = SpectrumAnalyzer()
        y = np.random.rand(MINIMUM_SAMPLE_RATE)  # 1 second at minimum rate
        assert analyzer._validate_audio(y, MINIMUM_SAMPLE_RATE) is True

    def test_extract_band_features_shape(self) -> None:
        """Test that extracted features have correct shape."""
        analyzer = SpectrumAnalyzer()

        # Create synthetic PSD data
        freqs = np.linspace(0, 22050, 4097)  # From FFT
        psd = np.random.rand(4097)

        features = analyzer._extract_band_features(psd, freqs)

        assert features is not None
        assert features.shape == (150,)
        assert features.dtype == np.float32

    def test_extract_band_features_normalized(self) -> None:
        """Test that features are normalized to 0-1 range."""
        analyzer = SpectrumAnalyzer()

        freqs = np.linspace(0, 22050, 4097)
        psd = np.random.rand(4097) * 1000  # Large values

        features = analyzer._extract_band_features(psd, freqs)

        assert features is not None
        assert np.all(features >= 0)
        assert np.all(features <= 1)


class TestSpectrumAnalyzerIsVbr:
    """Tests for SpectrumAnalyzer accepting and propagating is_vbr metadata."""

    def test_analyze_file_accepts_is_vbr_parameter(self, tmp_path, monkeypatch) -> None:
        """SpectrumAnalyzer.analyze_file should accept is_vbr parameter."""
        analyzer = SpectrumAnalyzer()

        # Mock librosa.load to return valid audio data
        def mock_load(file_path, sr=None, mono=True):
            return np.random.rand(44100), 44100  # 1 second at 44.1kHz

        monkeypatch.setattr("beetsplug.bitrater.spectrum.librosa.load", mock_load)

        # Create a dummy file
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # Should accept is_vbr parameter without error
        result = analyzer.analyze_file(str(test_file), is_vbr=1.0)

        assert result is not None
        assert result.is_vbr == 1.0

    def test_analyze_file_is_vbr_defaults_to_zero(self, tmp_path, monkeypatch) -> None:
        """SpectrumAnalyzer.analyze_file should default is_vbr to 0.0."""
        analyzer = SpectrumAnalyzer()

        def mock_load(file_path, sr=None, mono=True):
            return np.random.rand(44100), 44100

        monkeypatch.setattr("beetsplug.bitrater.spectrum.librosa.load", mock_load)

        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # Without is_vbr parameter, should default to 0.0
        result = analyzer.analyze_file(str(test_file))

        assert result is not None
        assert result.is_vbr == 0.0

    def test_analyze_file_propagates_is_vbr_cbr(self, tmp_path, monkeypatch) -> None:
        """SpectrumAnalyzer should propagate is_vbr=0.0 for CBR files."""
        analyzer = SpectrumAnalyzer()

        def mock_load(file_path, sr=None, mono=True):
            return np.random.rand(44100), 44100

        monkeypatch.setattr("beetsplug.bitrater.spectrum.librosa.load", mock_load)

        test_file = tmp_path / "cbr_192.mp3"
        test_file.touch()

        result = analyzer.analyze_file(str(test_file), is_vbr=0.0)

        assert result is not None
        assert result.is_vbr == 0.0

    def test_analyze_file_propagates_is_vbr_vbr(self, tmp_path, monkeypatch) -> None:
        """SpectrumAnalyzer should propagate is_vbr=1.0 for VBR files."""
        analyzer = SpectrumAnalyzer()

        def mock_load(file_path, sr=None, mono=True):
            return np.random.rand(44100), 44100

        monkeypatch.setattr("beetsplug.bitrater.spectrum.librosa.load", mock_load)

        test_file = tmp_path / "vbr_v0.mp3"
        test_file.touch()

        result = analyzer.analyze_file(str(test_file), is_vbr=1.0)

        assert result is not None
        assert result.is_vbr == 1.0


class TestSpectralFeaturesIsVbr:
    """Tests for is_vbr field in SpectralFeatures for VBR/CBR discrimination."""

    def test_spectral_features_has_is_vbr_field(self) -> None:
        """SpectralFeatures should have is_vbr field for VBR/CBR metadata."""
        features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000.0, 16040.0)] * 150,
            is_vbr=1.0,
        )
        assert hasattr(features, "is_vbr")
        assert features.is_vbr == 1.0

    def test_spectral_features_is_vbr_defaults_to_zero(self) -> None:
        """SpectralFeatures.is_vbr should default to 0.0 (CBR/unknown)."""
        features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000.0, 16040.0)] * 150,
        )
        assert features.is_vbr == 0.0

    def test_spectral_features_is_vbr_accepts_float(self) -> None:
        """is_vbr field should accept float values 0.0 or 1.0."""
        # VBR file
        vbr_features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000.0, 16040.0)] * 150,
            is_vbr=1.0,
        )
        assert vbr_features.is_vbr == 1.0

        # CBR file
        cbr_features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000.0, 16040.0)] * 150,
            is_vbr=0.0,
        )
        assert cbr_features.is_vbr == 0.0


class TestUltrasonicFeatures:
    """Tests for ultrasonic feature extraction."""

    def test_spectral_features_has_ultrasonic_field(self):
        """SpectralFeatures should have ultrasonic_features field."""
        features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000, 16040)] * 150,
        )
        assert hasattr(features, 'ultrasonic_features')
        assert features.ultrasonic_features.shape == (4,)

    def test_as_vector_includes_ultrasonic_features(self):
        """as_vector should include ultrasonic_features in output."""
        features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000, 16040)] * 150,
            ultrasonic_features=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )
        vector = features.as_vector()
        # 150 + 6 + 8 + 6 + 4 + 1 = 175 features
        assert vector.shape == (175,)
        # Ultrasonic features should be at position 170-173 (before is_vbr)
        assert vector[170] == 1.0
        assert vector[171] == 2.0
        assert vector[172] == 3.0
        assert vector[173] == 4.0


class TestExtractUltrasonicFeatures:
    """Tests for _extract_ultrasonic_features method."""

    @pytest.fixture
    def analyzer(self):
        """Create SpectrumAnalyzer with no cache."""
        return SpectrumAnalyzer(cache_dir=None)

    def test_returns_four_features(self, analyzer):
        """Should return exactly 4 features."""
        # Create mock PSD with frequencies up to 22050 Hz
        freqs = np.linspace(0, 22050, 2048)
        psd = np.ones_like(freqs) * 1e-6  # Flat spectrum

        result = analyzer._extract_ultrasonic_features(psd, freqs)

        assert result.shape == (4,)
        assert result.dtype == np.float32

    def test_v0_like_spectrum_high_shelf_flatness(self, analyzer):
        """V0-like spectrum (hard cutoff) should have high shelf flatness."""
        freqs = np.linspace(0, 22050, 2048)
        psd = np.ones_like(freqs) * 1e-3
        # Simulate V0 cutoff: drop to noise floor above 20kHz
        psd[freqs > 20000] = 1e-10  # Flat noise floor

        result = analyzer._extract_ultrasonic_features(psd, freqs)

        # ultrasonic_variance should be low (flat noise floor)
        assert result[0] < 1.0  # Low variance
        # energy_ratio should be very low (hard shelf drop)
        assert result[1] < 0.01
        # shelf_flatness should be high (uniform noise floor)
        assert result[3] > 0.5

    def test_lossless_like_spectrum_low_shelf_flatness(self, analyzer):
        """Lossless-like spectrum (content above 20kHz) should have low shelf flatness."""
        freqs = np.linspace(0, 22050, 2048)
        psd = np.ones_like(freqs) * 1e-3
        # Simulate lossless: content continues above 20kHz with variation
        psd[freqs > 20000] = 1e-4 + np.random.random(np.sum(freqs > 20000)) * 1e-4

        result = analyzer._extract_ultrasonic_features(psd, freqs)

        # ultrasonic_variance should be higher (varying content)
        assert result[0] > 0.1  # Higher variance
        # energy_ratio should be higher (content continues)
        assert result[1] > 0.05
        # shelf_flatness should be lower (not uniform)
        assert result[3] < 0.8

    def test_handles_empty_frequency_range(self, analyzer):
        """Should handle edge case of missing frequency data gracefully."""
        freqs = np.linspace(0, 15000, 1024)  # No ultrasonic data
        psd = np.ones_like(freqs) * 1e-6

        result = analyzer._extract_ultrasonic_features(psd, freqs)

        assert result.shape == (4,)
        # Should return zeros for missing data
        assert np.allclose(result, 0.0)

    def test_analyze_file_includes_ultrasonic_features(self, tmp_path, monkeypatch):
        """analyze_file should populate ultrasonic_features."""
        analyzer = SpectrumAnalyzer(cache_dir=None)

        # Mock librosa.load to return valid audio data
        def mock_load(file_path, sr=None, mono=True):
            return np.random.rand(44100), 44100  # 1 second at 44.1kHz

        monkeypatch.setattr("beetsplug.bitrater.spectrum.librosa.load", mock_load)

        # Create a dummy file
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        result = analyzer.analyze_file(str(test_file))

        assert result is not None
        assert hasattr(result, 'ultrasonic_features')
        assert result.ultrasonic_features.shape == (4,)
        # Should have some values (may be zero for simple random signal)
        assert result.ultrasonic_features.dtype == np.float32
