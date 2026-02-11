"""Additional tests for spectrum.py to improve coverage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bitrater.constants import SPECTRAL_PARAMS
from bitrater.spectrum import SpectrumAnalyzer


@pytest.fixture
def analyzer(tmp_path):
    """Create a SpectrumAnalyzer with temp cache dir."""
    return SpectrumAnalyzer(cache_dir=tmp_path / "cache")


class TestExtractBandFeatures:
    """Tests for _extract_band_features."""

    def test_returns_correct_shape(self, analyzer):
        freqs = np.linspace(0, 22050, 4097)
        psd = np.random.rand(4097)

        result = analyzer._extract_band_features(psd, freqs)
        assert result is not None
        assert result.shape == (SPECTRAL_PARAMS["num_bands"],)

    def test_returns_none_insufficient_resolution(self, analyzer):
        # Very few frequency points
        freqs = np.linspace(16000, 22050, 10)
        psd = np.random.rand(10)

        result = analyzer._extract_band_features(psd, freqs)
        assert result is None

    def test_normalized_range(self, analyzer):
        freqs = np.linspace(0, 22050, 4097)
        psd = np.random.rand(4097) * 100

        result = analyzer._extract_band_features(psd, freqs)
        assert result is not None
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_flat_spectrum(self, analyzer):
        """Flat spectrum should produce uniform features."""
        freqs = np.linspace(0, 22050, 4097)
        psd = np.ones(4097) * 0.5

        result = analyzer._extract_band_features(psd, freqs)
        assert result is not None


class TestExtractCutoffFeatures:
    """Tests for _extract_cutoff_features."""

    def test_returns_six_features(self, analyzer):
        freqs = np.linspace(0, 22050, 4097)
        psd = np.random.rand(4097)

        result = analyzer._extract_cutoff_features(psd, freqs)
        assert len(result) == 6

    def test_empty_band_returns_zeros(self, analyzer):
        # Frequencies all below our analysis range
        freqs = np.linspace(0, 10000, 1000)
        psd = np.random.rand(1000)

        result = analyzer._extract_cutoff_features(psd, freqs)
        np.testing.assert_array_equal(result, np.zeros(6, dtype=np.float32))


class TestExtractTemporalFeatures:
    """Tests for _extract_temporal_features."""

    def test_returns_eight_features(self, analyzer):
        y = np.random.rand(44100).astype(np.float32)  # 1 second
        result = analyzer._extract_temporal_features(y, 44100)
        assert len(result) == 8

    def test_zero_windows(self, analyzer):
        y = np.random.rand(44100).astype(np.float32)
        result = analyzer._extract_temporal_features(y, 44100, n_windows=0)
        np.testing.assert_array_equal(result, np.zeros(8, dtype=np.float32))

    def test_single_window(self, analyzer):
        y = np.random.rand(44100).astype(np.float32)
        result = analyzer._extract_temporal_features(y, 44100, n_windows=1)
        assert len(result) == 8


class TestExtractArtifactFeatures:
    """Tests for _extract_artifact_features."""

    def test_returns_six_features(self, analyzer):
        freqs = np.linspace(0, 22050, 4097)
        psd = np.random.rand(4097)
        cutoff_feats = np.array([0.5, 3, 0.3, 0.1, 0.2, 0.1], dtype=np.float32)

        result = analyzer._extract_artifact_features(psd, freqs, cutoff_feats)
        assert len(result) == 6

    def test_empty_band_returns_zeros(self, analyzer):
        freqs = np.linspace(0, 10000, 1000)
        psd = np.random.rand(1000)
        cutoff_feats = np.zeros(6, dtype=np.float32)

        result = analyzer._extract_artifact_features(psd, freqs, cutoff_feats)
        np.testing.assert_array_equal(result, np.zeros(6, dtype=np.float32))


class TestSpectralFlatness:
    """Tests for _spectral_flatness."""

    def test_constant_signal_is_one(self, analyzer):
        values = np.ones(100) * -10.0  # Constant dB values
        result = analyzer._spectral_flatness(values)
        assert abs(result - 1.0) < 0.01

    def test_empty_returns_zero(self, analyzer):
        result = analyzer._spectral_flatness(np.array([]))
        assert result == 0.0

    def test_values_with_nan(self, analyzer):
        values = np.array([1.0, float("nan"), 2.0])
        result = analyzer._spectral_flatness(values)
        assert np.isfinite(result)


class TestEstimateCutoffNormalized:
    """Tests for _estimate_cutoff_normalized."""

    def test_full_spectrum(self, analyzer):
        freqs = np.linspace(0, 22050, 4097)
        psd = np.ones(4097) * 0.5  # Flat spectrum

        result = analyzer._estimate_cutoff_normalized(freqs, psd)
        assert 0 <= result <= 1

    def test_empty_band(self, analyzer):
        freqs = np.linspace(0, 10000, 1000)
        psd = np.random.rand(1000)

        result = analyzer._estimate_cutoff_normalized(freqs, psd)
        assert result == 0.0


class TestSplitFeatureVector:
    """Tests for _split_feature_vector."""

    def test_roundtrip(self, analyzer):
        # Create a combined vector
        psd = np.random.rand(150).astype(np.float32)
        cutoff = np.random.rand(6).astype(np.float32)
        temporal = np.random.rand(8).astype(np.float32)
        artifact = np.random.rand(6).astype(np.float32)
        sfb21 = np.random.rand(6).astype(np.float32)
        rolloff = np.random.rand(4).astype(np.float32)

        combined = np.concatenate([psd, cutoff, temporal, artifact, sfb21, rolloff])
        metadata = {
            "n_bands": 150,
            "cutoff_len": 6,
            "temporal_len": 8,
            "artifact_len": 6,
            "sfb21_len": 6,
            "rolloff_len": 4,
        }

        result = analyzer._split_feature_vector(combined, metadata)
        r_psd, r_cutoff, r_temporal, r_artifact, r_sfb21, r_rolloff = result

        np.testing.assert_array_almost_equal(r_psd, psd)
        np.testing.assert_array_almost_equal(r_cutoff, cutoff)
        np.testing.assert_array_almost_equal(r_temporal, temporal)
        np.testing.assert_array_almost_equal(r_artifact, artifact)
        np.testing.assert_array_almost_equal(r_sfb21, sfb21)
        np.testing.assert_array_almost_equal(r_rolloff, rolloff)

    def test_missing_metadata_uses_defaults(self, analyzer):
        combined = np.random.rand(180).astype(np.float32)
        metadata = {"n_bands": 150}

        result = analyzer._split_feature_vector(combined, metadata)
        assert len(result) == 6


class TestGetPsd:
    """Tests for get_psd."""

    def test_returns_cached_psd(self, analyzer):
        psd = np.random.rand(100)
        freqs = np.linspace(0, 22050, 100)

        analyzer._last_psd_path = "/test/file.mp3"
        analyzer._last_psd = (psd, freqs)

        result = analyzer.get_psd("/test/file.mp3")
        assert result is not None
        np.testing.assert_array_equal(result[0], psd)

    def test_nonexistent_file_returns_none(self, analyzer):
        result = analyzer.get_psd("/nonexistent/file.mp3")
        assert result is None


class TestClearCache:
    """Tests for clear_cache."""

    def test_delegates_to_feature_cache(self, analyzer):
        analyzer.cache = MagicMock()
        analyzer.clear_cache()
        analyzer.cache.clear.assert_called_once()


class TestAnalyzeFileWithCache:
    """Tests for analyze_file cache interactions."""

    def test_returns_cached_features(self, analyzer):
        """When cache has valid features, should return them without loading audio."""
        num_bands = SPECTRAL_PARAMS["num_bands"]
        psd = np.random.rand(num_bands).astype(np.float32)
        cutoff = np.random.rand(6).astype(np.float32)
        temporal = np.random.rand(8).astype(np.float32)
        artifact = np.random.rand(6).astype(np.float32)
        sfb21 = np.random.rand(6).astype(np.float32)
        rolloff = np.random.rand(4).astype(np.float32)

        combined = np.concatenate([psd, cutoff, temporal, artifact, sfb21, rolloff])
        metadata = {
            "n_bands": num_bands,
            "approach": "encoder_agnostic_v8",
            "cutoff_len": 6,
            "temporal_len": 8,
            "artifact_len": 6,
            "sfb21_len": 6,
            "rolloff_len": 4,
            "band_frequencies": analyzer._band_frequencies,
        }

        analyzer.cache = MagicMock()
        analyzer.cache.get_features.return_value = (combined, metadata)

        # Should NOT call librosa.load
        with patch("bitrater.spectrum.librosa.load") as mock_load:
            result = analyzer.analyze_file("/some/file.mp3", is_vbr=1.0)
            mock_load.assert_not_called()

        assert result is not None
        assert result.is_vbr == 1.0

    def test_cache_miss_wrong_approach(self, analyzer):
        """Cache with different approach version should be treated as miss."""
        num_bands = SPECTRAL_PARAMS["num_bands"]
        combined = np.random.rand(num_bands + 30).astype(np.float32)
        metadata = {
            "n_bands": num_bands,
            "approach": "old_approach_v1",  # Wrong approach
        }

        analyzer.cache = MagicMock()
        analyzer.cache.get_features.return_value = (combined, metadata)

        # Will try to load audio - should fail since file doesn't exist
        result = analyzer.analyze_file("/nonexistent/file.mp3")
        assert result is None
