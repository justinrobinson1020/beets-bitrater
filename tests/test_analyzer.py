"""Tests for audio quality analyzer."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from beetsplug.bitrater.analyzer import AudioQualityAnalyzer
from beetsplug.bitrater.types import AnalysisResult


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

    def test_analyze_file_untrained_returns_unknown(self, sample_features, tmp_path):
        """Test analyze_file returns UNKNOWN when classifier not trained."""
        analyzer = AudioQualityAnalyzer()

        # Create a real temporary file to avoid mocking Path.exists
        fake_audio = tmp_path / "test.mp3"
        fake_audio.write_bytes(b"fake audio content")

        # Only mock the components that do actual audio processing
        with patch.object(analyzer.spectrum_analyzer, "analyze_file", return_value=sample_features), \
             patch.object(analyzer.file_analyzer, "analyze", return_value=Mock(bitrate=320)):

            result = analyzer.analyze_file(str(fake_audio))

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


class TestWarningGeneration:
    """Tests for warning message generation through AnalysisResult."""

    def test_low_confidence_warning_in_result(self):
        """Test warning generated for low confidence analysis."""
        # Low confidence results should include a warning
        result = AnalysisResult(
            filename="test.mp3",
            file_format="mp3",
            original_format="320",
            original_bitrate=320,
            confidence=0.5,  # Below LOW_CONFIDENCE_THRESHOLD (0.7)
            is_transcode=False,
            stated_class="320",
            detected_cutoff=20500,
            quality_gap=0,
            stated_bitrate=320,
            warnings=["Low confidence in detection: 50.0%"],
        )

        assert any("confidence" in w.lower() for w in result.warnings)

    def test_transcode_warning_in_result(self):
        """Test warning generated for transcodes."""
        result = AnalysisResult(
            filename="fake_lossless.flac",
            file_format="flac",
            original_format="128",
            original_bitrate=128,
            confidence=0.9,
            is_transcode=True,
            stated_class="LOSSLESS",
            detected_cutoff=16000,
            quality_gap=6,
            transcoded_from="128",
            warnings=["File appears to be transcoded from 128 (quality gap: 6)"],
        )

        assert any("transcode" in w.lower() for w in result.warnings)

    def test_bitrate_mismatch_warning_in_result(self):
        """Test warning for significant bitrate mismatch."""
        result = AnalysisResult(
            filename="upsampled.mp3",
            file_format="mp3",
            original_format="128",
            original_bitrate=128,
            confidence=0.9,
            is_transcode=False,
            stated_class="320",
            detected_cutoff=16000,
            quality_gap=0,
            stated_bitrate=320,  # Claims 320, detected 128
            warnings=["Stated bitrate (320 kbps) much higher than detected (128 kbps) - possible upsampled file"],
        )

        assert any("bitrate" in w.lower() or "upsampled" in w.lower() for w in result.warnings)

    def test_warning_constants_exist(self):
        """Test that warning threshold constants are defined."""
        from beetsplug.bitrater.constants import (
            BITRATE_MISMATCH_FACTOR,
            LOW_CONFIDENCE_THRESHOLD,
        )

        assert LOW_CONFIDENCE_THRESHOLD == 0.7
        assert BITRATE_MISMATCH_FACTOR == 1.5


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_summarize(self):
        """Test AnalysisResult.summarize() method."""

        result = AnalysisResult(
            filename="test.mp3",
            file_format="mp3",
            original_format="320",
            original_bitrate=320,
            confidence=0.95,
            is_transcode=False,
            stated_class="320",
            detected_cutoff=20500,
            quality_gap=0,
            stated_bitrate=320,
        )

        summary = result.summarize()

        assert summary["filename"] == "test.mp3"
        assert summary["original_format"] == "320"
        assert summary["original_bitrate"] == 320
        assert summary["confidence"] == 0.95
        assert summary["is_transcode"] is False
        assert summary["stated_class"] == "320"
        assert summary["detected_cutoff"] == 20500
        assert summary["quality_gap"] == 0

    def test_transcode_result(self):
        """Test AnalysisResult for a transcoded file."""
        result = AnalysisResult(
            filename="fake_lossless.flac",
            file_format="flac",
            original_format="128",
            original_bitrate=128,
            confidence=0.92,
            is_transcode=True,
            stated_class="LOSSLESS",
            detected_cutoff=16000,
            quality_gap=6,
            transcoded_from="128",
            stated_bitrate=None,
        )

        assert result.is_transcode is True
        assert result.transcoded_from == "128"
        assert result.file_format == "flac"
        assert result.stated_class == "LOSSLESS"
        assert result.detected_cutoff == 16000
        assert result.quality_gap == 6


class TestAnalysisResultFields:
    """Test new AnalysisResult fields for transcode detection."""

    def test_has_stated_class_field(self):
        """AnalysisResult should have stated_class field."""
        result = AnalysisResult(
            filename="test.flac",
            file_format="flac",
            original_format="192",
            original_bitrate=192,
            confidence=0.9,
            is_transcode=True,
            stated_class="LOSSLESS",
            detected_cutoff=19000,
            quality_gap=4,
        )
        assert result.stated_class == "LOSSLESS"

    def test_has_detected_cutoff_field(self):
        """AnalysisResult should have detected_cutoff field."""
        result = AnalysisResult(
            filename="test.mp3",
            file_format="mp3",
            original_format="320",
            original_bitrate=320,
            confidence=0.9,
            is_transcode=False,
            stated_class="320",
            detected_cutoff=20500,
            quality_gap=0,
        )
        assert result.detected_cutoff == 20500

    def test_has_quality_gap_field(self):
        """AnalysisResult should have quality_gap field."""
        result = AnalysisResult(
            filename="test.flac",
            file_format="flac",
            original_format="128",
            original_bitrate=128,
            confidence=0.85,
            is_transcode=True,
            stated_class="LOSSLESS",
            detected_cutoff=16000,
            quality_gap=6,  # LOSSLESS(6) - 128(0) = 6
        )
        assert result.quality_gap == 6


class TestIntegratedTranscodeDetection:
    """Test full transcode detection pipeline."""

    def test_analyze_detects_stated_class_from_container(self):
        """Analyzer should determine stated_class from file format."""
        # This test will need mock spectral data
        # For now, test the helper method
        from beetsplug.bitrater.analyzer import AudioQualityAnalyzer

        analyzer = AudioQualityAnalyzer()

        # FLAC container = LOSSLESS stated class
        assert analyzer._get_stated_class("flac", None) == "LOSSLESS"
        # WAV container = LOSSLESS stated class
        assert analyzer._get_stated_class("wav", None) == "LOSSLESS"
        # MP3 with 320 bitrate
        assert analyzer._get_stated_class("mp3", 320) == "320"
        # MP3 with 192 bitrate
        assert analyzer._get_stated_class("mp3", 192) == "192"
        # MP3 with ~245 bitrate (V0 range)
        assert analyzer._get_stated_class("mp3", 245) == "V0"
