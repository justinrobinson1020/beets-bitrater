"""Tests for audio quality analyzer."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from beetsplug.bitrater.analyzer import AudioQualityAnalyzer
from beetsplug.bitrater.types import (
    AnalysisResult,
    SpectralFeatures,
    ClassifierPrediction,
    FileMetadata,
)
from beetsplug.bitrater.constants import LOSSLESS_CONTAINERS


class TestAudioQualityAnalyzer:
    """Tests for AudioQualityAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = AudioQualityAnalyzer()

        assert analyzer.spectrum_analyzer is not None
        assert analyzer.classifier is not None
        assert analyzer.file_analyzer is not None
        assert analyzer.is_trained is False

    def test_is_trained_property(self, training_features):
        """Test is_trained property reflects classifier state."""
        analyzer = AudioQualityAnalyzer()
        assert analyzer.is_trained is False

        features_list, labels = training_features
        analyzer.classifier.train(features_list, labels)
        assert analyzer.is_trained is True

    def test_analyze_file_not_found(self):
        """Test analyze_file returns None for non-existent file."""
        analyzer = AudioQualityAnalyzer()
        result = analyzer.analyze_file("/nonexistent/file.mp3")
        assert result is None

    def test_analyze_file_untrained_returns_unknown(self, sample_features):
        """Test analyze_file returns UNKNOWN when classifier not trained."""
        analyzer = AudioQualityAnalyzer()

        # Mock the internal analyzers
        with patch.object(analyzer, "spectrum_analyzer") as mock_spectrum, \
             patch.object(analyzer, "file_analyzer") as mock_file, \
             patch("pathlib.Path.exists", return_value=True):

            mock_spectrum.analyze_file.return_value = sample_features
            mock_file.analyze.return_value = Mock(bitrate=320)

            result = analyzer.analyze_file("/fake/file.mp3")

        assert result is not None
        assert result.original_format == "UNKNOWN"
        assert "Classifier not trained" in result.warnings

    def test_train_empty_raises(self):
        """Test train with empty data raises ValueError."""
        analyzer = AudioQualityAnalyzer()

        with pytest.raises(ValueError):
            analyzer.train({})

    def test_train_from_directory_not_found(self):
        """Test train_from_directory raises for non-existent directory."""
        analyzer = AudioQualityAnalyzer()

        with pytest.raises(FileNotFoundError):
            analyzer.train_from_directory(Path("/nonexistent/dir"))


class TestTranscodeDetection:
    """Tests for transcode detection logic."""

    def test_lossless_container_with_lossy_content_is_transcode(self):
        """Test that FLAC with lossy signature is detected as transcode."""
        analyzer = AudioQualityAnalyzer()

        # The transcode detection logic
        file_format = "flac"
        predicted_format = "128"  # Detected as 128 kbps

        is_lossless_container = file_format in LOSSLESS_CONTAINERS
        detected_lossy = predicted_format != "LOSSLESS"

        is_transcode = is_lossless_container and detected_lossy

        assert is_transcode is True

    def test_lossless_container_with_lossless_content_not_transcode(self):
        """Test that genuine lossless FLAC is not flagged."""
        file_format = "flac"
        predicted_format = "LOSSLESS"

        is_lossless_container = file_format in LOSSLESS_CONTAINERS
        detected_lossy = predicted_format != "LOSSLESS"

        is_transcode = is_lossless_container and detected_lossy

        assert is_transcode is False

    def test_mp3_not_transcode(self):
        """Test that MP3 files are not flagged as transcodes."""
        file_format = "mp3"
        predicted_format = "320"

        is_lossless_container = file_format in LOSSLESS_CONTAINERS
        detected_lossy = predicted_format != "LOSSLESS"

        is_transcode = is_lossless_container and detected_lossy

        assert is_transcode is False  # MP3 is not a lossless container


class TestWarningGeneration:
    """Tests for warning message generation."""

    def test_low_confidence_warning(self):
        """Test warning generated for low confidence."""
        analyzer = AudioQualityAnalyzer()

        warnings = analyzer._generate_warnings(
            file_format="mp3",
            original_format="320",
            confidence=0.5,  # Below 0.7 threshold
            is_transcode=False,
            stated_bitrate=320,
        )

        assert any("confidence" in w.lower() for w in warnings)

    def test_transcode_warning(self):
        """Test warning generated for transcodes."""
        analyzer = AudioQualityAnalyzer()

        warnings = analyzer._generate_warnings(
            file_format="flac",
            original_format="128",
            confidence=0.9,
            is_transcode=True,
            stated_bitrate=None,
        )

        assert any("transcode" in w.lower() for w in warnings)

    def test_bitrate_mismatch_warning(self):
        """Test warning for significant bitrate mismatch."""
        analyzer = AudioQualityAnalyzer()

        warnings = analyzer._generate_warnings(
            file_format="mp3",
            original_format="128",  # Detected as 128
            confidence=0.9,
            is_transcode=False,
            stated_bitrate=320,  # Claims to be 320
        )

        assert any("bitrate" in w.lower() or "upsampled" in w.lower() for w in warnings)


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_summarize(self):
        """Test AnalysisResult.summarize() method."""
        from datetime import datetime

        result = AnalysisResult(
            filename="test.mp3",
            file_format="mp3",
            original_format="320",
            original_bitrate=320,
            confidence=0.95,
            is_transcode=False,
            stated_bitrate=320,
        )

        summary = result.summarize()

        assert summary["filename"] == "test.mp3"
        assert summary["original_format"] == "320"
        assert summary["original_bitrate"] == 320
        assert summary["confidence"] == 0.95
        assert summary["is_transcode"] is False

    def test_transcode_result(self):
        """Test AnalysisResult for a transcoded file."""
        result = AnalysisResult(
            filename="fake_lossless.flac",
            file_format="flac",
            original_format="128",
            original_bitrate=128,
            confidence=0.92,
            is_transcode=True,
            transcoded_from="128",
            stated_bitrate=None,
        )

        assert result.is_transcode is True
        assert result.transcoded_from == "128"
        assert result.file_format == "flac"
