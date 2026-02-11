"""Tests for spectrum analyzer."""

import numpy as np
import pytest

from bitrater.constants import (
    MINIMUM_DURATION,
    MINIMUM_SAMPLE_RATE,
    SPECTRAL_PARAMS,
)
from bitrater.spectrum import SpectrumAnalyzer
from bitrater.types import SpectralFeatures


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

        monkeypatch.setattr("bitrater.spectrum.librosa.load", mock_load)

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

        monkeypatch.setattr("bitrater.spectrum.librosa.load", mock_load)

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

        monkeypatch.setattr("bitrater.spectrum.librosa.load", mock_load)

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

        monkeypatch.setattr("bitrater.spectrum.librosa.load", mock_load)

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


class TestSFB21AndRolloffFeatures:
    """Tests for SFB21 and rolloff feature fields in SpectralFeatures."""

    def test_spectral_features_has_sfb21_field(self) -> None:
        """SpectralFeatures should have sfb21_features field."""
        features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000, 16040)] * 150,
        )
        assert hasattr(features, "sfb21_features")
        assert features.sfb21_features.shape == (6,)

    def test_spectral_features_has_rolloff_field(self) -> None:
        """SpectralFeatures should have rolloff_features field."""
        features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000, 16040)] * 150,
        )
        assert hasattr(features, "rolloff_features")
        assert features.rolloff_features.shape == (4,)

    def test_as_vector_includes_sfb21_and_rolloff(self) -> None:
        """as_vector should include sfb21, rolloff, and discriminative features."""
        features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000, 16040)] * 150,
            sfb21_features=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            rolloff_features=np.array([7.0, 8.0, 9.0, 10.0], dtype=np.float32),
            discriminative_features=np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=np.float32),
        )
        vector = features.as_vector()
        # 150 PSD + 6 cutoff + 6 SFB21 + 4 rolloff + 6 discriminative + 1 is_vbr = 173
        assert vector.shape == (173,)
        # SFB21 features at position 156-161 (after cutoff)
        assert vector[156] == 1.0
        assert vector[161] == 6.0
        # Rolloff features at position 162-165
        assert vector[162] == 7.0
        assert vector[165] == 10.0
        # Discriminative features at position 166-171
        assert vector[166] == 11.0
        assert vector[171] == 16.0
        # is_vbr at the end
        assert vector[172] == 0.0


class TestExtractSFB21Features:
    """Tests for _extract_sfb21_features method."""

    @pytest.fixture
    def analyzer(self):
        """Create SpectrumAnalyzer with no cache."""
        return SpectrumAnalyzer(cache_dir=None)

    def test_returns_six_features(self, analyzer) -> None:
        """Should return exactly 6 features."""
        import librosa

        y = np.random.rand(44100).astype(np.float32)
        S_mag = np.abs(librosa.stft(y, n_fft=SPECTRAL_PARAMS["fft_size"]))
        freqs = librosa.fft_frequencies(sr=44100, n_fft=SPECTRAL_PARAMS["fft_size"])
        result = analyzer._extract_sfb21_features(S_mag, freqs)
        assert result.shape == (6,)
        assert result.dtype == np.float32

    def test_handles_empty_audio(self, analyzer) -> None:
        """Should return zeros for empty STFT."""
        S_mag = np.array([], dtype=np.float32)
        freqs = np.array([], dtype=np.float32)
        result = analyzer._extract_sfb21_features(S_mag, freqs)
        assert result.shape == (6,)
        assert np.allclose(result, 0.0)


class TestExtractRolloffFeatures:
    """Tests for _extract_rolloff_features method."""

    @pytest.fixture
    def analyzer(self):
        """Create SpectrumAnalyzer with no cache."""
        return SpectrumAnalyzer(cache_dir=None)

    def test_returns_four_features(self, analyzer) -> None:
        """Should return exactly 4 features."""
        import librosa

        y = np.random.rand(44100).astype(np.float32)
        S_power = np.abs(librosa.stft(y, n_fft=SPECTRAL_PARAMS["fft_size"])) ** 2
        freqs = librosa.fft_frequencies(sr=44100, n_fft=SPECTRAL_PARAMS["fft_size"])
        result = analyzer._extract_rolloff_features(S_power, freqs)
        assert result.shape == (4,)
        assert result.dtype == np.float32

    def test_handles_empty_audio(self, analyzer) -> None:
        """Should return zeros for empty STFT."""
        S_power = np.array([], dtype=np.float32)
        freqs = np.array([], dtype=np.float32)
        result = analyzer._extract_rolloff_features(S_power, freqs)
        assert result.shape == (4,)
        assert np.allclose(result, 0.0)

    def test_analyze_file_includes_sfb21_and_rolloff(self, tmp_path, monkeypatch) -> None:
        """analyze_file should populate sfb21 and rolloff features."""
        analyzer = SpectrumAnalyzer(cache_dir=None)

        def mock_load(file_path, sr=None, mono=True):
            return np.random.rand(44100), 44100

        monkeypatch.setattr("bitrater.spectrum.librosa.load", mock_load)

        test_file = tmp_path / "test.mp3"
        test_file.touch()

        result = analyzer.analyze_file(str(test_file))

        assert result is not None
        assert hasattr(result, "sfb21_features")
        assert result.sfb21_features.shape == (6,)
        assert result.sfb21_features.dtype == np.float32
        assert hasattr(result, "rolloff_features")
        assert result.rolloff_features.shape == (4,)
        assert result.rolloff_features.dtype == np.float32
