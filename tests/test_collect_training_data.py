"""Tests for _collect_training_data extracted from train/validate methods."""

from pathlib import Path

import pytest

from beetsplug.bitrater.analyzer import AudioQualityAnalyzer
from beetsplug.bitrater.constants import CLASS_LABELS


class TestCollectTrainingData:
    """Tests for _collect_training_data helper method."""

    def test_method_exists(self):
        """_collect_training_data should be a method on AudioQualityAnalyzer."""
        analyzer = AudioQualityAnalyzer()
        assert hasattr(analyzer, '_collect_training_data')
        assert callable(analyzer._collect_training_data)

    def test_flat_structure(self, tmp_path):
        """Should collect files from flat directory structure."""
        # Create flat structure
        for name in ["128", "v2", "192", "v0", "256", "320"]:
            d = tmp_path / name
            d.mkdir()
            (d / f"test_{name}.mp3").touch()

        lossless = tmp_path / "lossless"
        lossless.mkdir()
        (lossless / "test_lossless.flac").touch()

        analyzer = AudioQualityAnalyzer()
        data = analyzer._collect_training_data(tmp_path)

        assert len(data) == 7  # One file per class
        # Verify labels are correct
        for path, label in data.items():
            assert label in range(7)

    def test_nested_structure(self, tmp_path):
        """Should collect files from nested encoded/lossy/ structure."""
        lossy_dir = tmp_path / "encoded" / "lossy"
        lossy_dir.mkdir(parents=True)

        for name in ["128", "v2", "192", "v0", "256", "320"]:
            d = lossy_dir / name
            d.mkdir()
            (d / f"test_{name}.mp3").touch()

        lossless = tmp_path / "lossless"
        lossless.mkdir()
        (lossless / "test_lossless.flac").touch()

        analyzer = AudioQualityAnalyzer()
        data = analyzer._collect_training_data(tmp_path)

        assert len(data) == 7

    def test_empty_dir_raises(self, tmp_path):
        """Should raise ValueError if no training files found."""
        analyzer = AudioQualityAnalyzer()
        with pytest.raises(ValueError, match="No training files found"):
            analyzer._collect_training_data(tmp_path)

    def test_nonexistent_dir_raises(self, tmp_path):
        """Should raise FileNotFoundError for nonexistent directory."""
        analyzer = AudioQualityAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer._collect_training_data(tmp_path / "nonexistent")

    def test_returns_correct_labels(self, tmp_path):
        """Returned labels should match CLASS_LABELS constants."""
        d128 = tmp_path / "128"
        d128.mkdir()
        (d128 / "test.mp3").touch()

        lossless = tmp_path / "lossless"
        lossless.mkdir()
        (lossless / "test.flac").touch()

        analyzer = AudioQualityAnalyzer()
        data = analyzer._collect_training_data(tmp_path)

        labels = set(data.values())
        assert CLASS_LABELS["128"] in labels
        assert CLASS_LABELS["LOSSLESS"] in labels

    def test_filters_audio_extensions(self, tmp_path):
        """Should only include audio files, not other file types."""
        d128 = tmp_path / "128"
        d128.mkdir()
        (d128 / "test.mp3").touch()
        (d128 / "readme.txt").touch()
        (d128 / "cover.jpg").touch()

        analyzer = AudioQualityAnalyzer()
        data = analyzer._collect_training_data(tmp_path)

        assert len(data) == 1  # Only the mp3
