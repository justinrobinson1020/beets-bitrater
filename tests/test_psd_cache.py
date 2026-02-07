"""Tests for PSD caching in SpectrumAnalyzer to avoid double-load."""

import numpy as np
import pytest

from beetsplug.bitrater.spectrum import SpectrumAnalyzer


class TestPsdCache:
    """Tests for in-memory PSD caching between analyze_file and get_psd."""

    def test_get_psd_returns_cached_after_analyze_file(self, tmp_path, monkeypatch):
        """get_psd should return cached PSD if analyze_file was just called on same file."""
        analyzer = SpectrumAnalyzer(cache_dir=None)
        load_count = 0

        def mock_load(file_path, sr=None, mono=True):
            nonlocal load_count
            load_count += 1
            return np.random.rand(44100), 44100

        monkeypatch.setattr("beetsplug.bitrater.spectrum.librosa.load", mock_load)

        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # First call: analyze_file loads audio
        result = analyzer.analyze_file(str(test_file))
        assert result is not None
        first_load_count = load_count

        # Second call: get_psd should use cache, NOT reload
        psd_result = analyzer.get_psd(str(test_file))
        assert psd_result is not None
        assert load_count == first_load_count, "get_psd should not reload audio after analyze_file"

    def test_get_psd_cache_cleared_on_new_analyze(self, tmp_path, monkeypatch):
        """get_psd cache should not persist across different analyze_file calls."""
        analyzer = SpectrumAnalyzer(cache_dir=None)

        def mock_load(file_path, sr=None, mono=True):
            return np.random.rand(44100), 44100

        monkeypatch.setattr("beetsplug.bitrater.spectrum.librosa.load", mock_load)

        file1 = tmp_path / "file1.mp3"
        file1.touch()
        file2 = tmp_path / "file2.mp3"
        file2.touch()

        # Analyze file1
        analyzer.analyze_file(str(file1))

        # Analyze file2 — should clear file1's PSD cache
        analyzer.analyze_file(str(file2))

        # get_psd for file1 should NOT use stale cache
        # (it will reload from disk, which is correct)
        assert analyzer._last_psd_path != str(file1)

    def test_get_psd_returns_tuple(self, tmp_path, monkeypatch):
        """get_psd should return (psd, freqs) tuple from cache."""
        analyzer = SpectrumAnalyzer(cache_dir=None)

        def mock_load(file_path, sr=None, mono=True):
            return np.random.rand(44100), 44100

        monkeypatch.setattr("beetsplug.bitrater.spectrum.librosa.load", mock_load)

        test_file = tmp_path / "test.mp3"
        test_file.touch()

        analyzer.analyze_file(str(test_file))
        psd_result = analyzer.get_psd(str(test_file))

        assert psd_result is not None
        psd, freqs = psd_result
        assert isinstance(psd, np.ndarray)
        assert isinstance(freqs, np.ndarray)
