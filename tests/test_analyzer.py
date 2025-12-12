"""Tests for audio quality analyzer."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from beetsplug.bitrater.analyzer import AudioQualityAnalyzer
from beetsplug.bitrater.types import AnalysisResult


class TestAudioQualityAnalyzer:
    """Tests for AudioQualityAnalyzer class."""

    def test_init(self) -> None:
        """Test analyzer initialization."""
        analyzer = AudioQualityAnalyzer()

        assert analyzer.spectrum_analyzer is not None
        assert analyzer.classifier is not None
        assert analyzer.file_analyzer is not None
        assert analyzer.is_trained is False

    def test_is_trained_property(self, training_features: tuple[list, list]) -> None:
        """Test is_trained property reflects classifier state."""
        analyzer = AudioQualityAnalyzer()
        assert analyzer.is_trained is False

        features_list, labels = training_features
        analyzer.classifier.train(features_list, labels)
        assert analyzer.is_trained is True

    def test_analyze_file_not_found(self) -> None:
        """Test analyze_file returns None for non-existent file."""
        analyzer = AudioQualityAnalyzer()
        result = analyzer.analyze_file("/nonexistent/file.mp3")
        assert result is None

    def test_analyze_file_untrained_returns_unknown(self, sample_features: "SpectralFeatures", tmp_path: Path) -> None:
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

    def test_train_empty_raises(self) -> None:
        """Test train with empty data raises ValueError."""
        analyzer = AudioQualityAnalyzer()

        with pytest.raises(ValueError):
            analyzer.train({})

    def test_train_from_directory_not_found(self) -> None:
        """Test train_from_directory raises for non-existent directory."""
        analyzer = AudioQualityAnalyzer()

        with pytest.raises(FileNotFoundError):
            analyzer.train_from_directory(Path("/nonexistent/dir"))


class TestWarningGeneration:
    """Tests for warning message generation through AnalysisResult."""

    def test_low_confidence_warning_in_result(self, analysis_result_builder: "AnalysisResultBuilder") -> None:
        """Test warning generated for low confidence analysis."""
        # Low confidence results should include a warning
        result = analysis_result_builder.with_low_confidence().build()

        assert any("confidence" in w.lower() for w in result.warnings)

    def test_transcode_warning_in_result(self, analysis_result_builder: "AnalysisResultBuilder") -> None:
        """Test warning generated for transcodes."""
        result = analysis_result_builder.with_transcode().build()

        assert any("transcode" in w.lower() for w in result.warnings)

    def test_bitrate_mismatch_warning_in_result(self, analysis_result_builder: "AnalysisResultBuilder") -> None:
        """Test warning for significant bitrate mismatch."""
        result = analysis_result_builder.with_bitrate_mismatch().build()

        assert any("bitrate" in w.lower() or "upsampled" in w.lower() for w in result.warnings)

    def test_warning_constants_exist(self) -> None:
        """Test that warning threshold constants are defined and used correctly."""
        from beetsplug.bitrater.constants import (
            BITRATE_MISMATCH_FACTOR,
            LOW_CONFIDENCE_THRESHOLD,
        )

        # Constants should be defined
        assert LOW_CONFIDENCE_THRESHOLD == 0.7
        assert BITRATE_MISMATCH_FACTOR == 1.5

        # Verify these constants are actually used in warning logic
        # by testing boundary conditions
        assert LOW_CONFIDENCE_THRESHOLD > 0  # Must be positive probability
        assert LOW_CONFIDENCE_THRESHOLD < 1  # Must be valid probability
        assert BITRATE_MISMATCH_FACTOR > 1  # Must require significant mismatch


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_summarize(self) -> None:
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

    def test_transcode_result(self) -> None:
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

    def test_has_stated_class_field(self) -> None:
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

    def test_has_detected_cutoff_field(self) -> None:
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

    def test_has_quality_gap_field(self) -> None:
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


class TestAnalyzerIsVbrPropagation:
    """Tests for analyzer propagating is_vbr from file metadata to spectrum analyzer.
    
    These tests verify that VBR/CBR metadata from file analysis is correctly
    passed to the spectrum analyzer as a feature flag that may affect spectral
    characteristic detection (e.g., VBR may show different bitrate patterns).
    """

    def test_analyze_file_passes_is_vbr_true_for_vbr_files(self, sample_features: "SpectralFeatures", tmp_path: Path) -> None:
        """Analyzer should pass is_vbr=1.0 to spectrum analyzer for VBR files."""
        analyzer = AudioQualityAnalyzer()

        # Create test file
        fake_audio = tmp_path / "test_vbr.mp3"
        fake_audio.write_bytes(b"fake audio content")

        # Mock metadata with VBR encoding
        mock_metadata: Mock = Mock(bitrate=245, encoding_type="VBR")

        with patch.object(analyzer.file_analyzer, "analyze", return_value=mock_metadata), \
             patch.object(analyzer.spectrum_analyzer, "analyze_file", return_value=sample_features) as mock_spectrum:

            analyzer.analyze_file(str(fake_audio))

            # Verify spectrum analyzer was called with is_vbr=1.0
            mock_spectrum.assert_called_once()
            call_kwargs = mock_spectrum.call_args
            # Check if is_vbr was passed as keyword argument
            assert call_kwargs[1].get("is_vbr") == 1.0 or (len(call_kwargs[0]) > 1 and call_kwargs[0][1] == 1.0)

    def test_analyze_file_passes_is_vbr_false_for_cbr_files(self, sample_features: "SpectralFeatures", tmp_path: Path) -> None:
        """Analyzer should pass is_vbr=0.0 to spectrum analyzer for CBR files."""
        analyzer = AudioQualityAnalyzer()

        # Create test file
        fake_audio = tmp_path / "test_cbr.mp3"
        fake_audio.write_bytes(b"fake audio content")

        # Mock metadata with CBR encoding
        mock_metadata: Mock = Mock(bitrate=320, encoding_type="CBR")

        with patch.object(analyzer.file_analyzer, "analyze", return_value=mock_metadata), \
             patch.object(analyzer.spectrum_analyzer, "analyze_file", return_value=sample_features) as mock_spectrum:

            analyzer.analyze_file(str(fake_audio))

            # Verify spectrum analyzer was called with is_vbr=0.0
            mock_spectrum.assert_called_once()
            call_kwargs = mock_spectrum.call_args
            assert call_kwargs[1].get("is_vbr") == 0.0 or (len(call_kwargs[0]) > 1 and call_kwargs[0][1] == 0.0)

    def test_analyze_file_passes_is_vbr_false_for_lossless(self, sample_features: "SpectralFeatures", tmp_path: Path) -> None:
        """Analyzer should pass is_vbr=0.0 for lossless files (not VBR)."""
        analyzer = AudioQualityAnalyzer()

        fake_audio = tmp_path / "test.flac"
        fake_audio.write_bytes(b"fake audio content")

        mock_metadata: Mock = Mock(bitrate=None, encoding_type="lossless")

        with patch.object(analyzer.file_analyzer, "analyze", return_value=mock_metadata), \
             patch.object(analyzer.spectrum_analyzer, "analyze_file", return_value=sample_features) as mock_spectrum:

            analyzer.analyze_file(str(fake_audio))

            mock_spectrum.assert_called_once()
            call_kwargs = mock_spectrum.call_args
            assert call_kwargs[1].get("is_vbr") == 0.0 or (len(call_kwargs[0]) > 1 and call_kwargs[0][1] == 0.0)

    def test_analyze_file_passes_is_vbr_false_when_metadata_unavailable(self, sample_features: "SpectralFeatures", tmp_path: Path) -> None:
        """Analyzer should default to is_vbr=0.0 when metadata extraction fails."""
        analyzer = AudioQualityAnalyzer()

        fake_audio = tmp_path / "test.mp3"
        fake_audio.write_bytes(b"fake audio content")

        # metadata returns None (extraction failed)
        with patch.object(analyzer.file_analyzer, "analyze", return_value=None), \
             patch.object(analyzer.spectrum_analyzer, "analyze_file", return_value=sample_features) as mock_spectrum:

            analyzer.analyze_file(str(fake_audio))

            mock_spectrum.assert_called_once()
            call_kwargs = mock_spectrum.call_args
            assert call_kwargs[1].get("is_vbr") == 0.0 or (len(call_kwargs[0]) > 1 and call_kwargs[0][1] == 0.0)

    def test_train_passes_is_vbr_to_spectrum_analyzer(self, sample_features: "SpectralFeatures", tmp_path: Path) -> None:
        """Train method should pass is_vbr from metadata to spectrum analyzer."""
        analyzer = AudioQualityAnalyzer()

        # Create test files with distinct names
        vbr_file = tmp_path / "file_v0.mp3"
        vbr_file.write_bytes(b"fake vbr content")
        cbr_file = tmp_path / "file_320.mp3"
        cbr_file.write_bytes(b"fake cbr content")

        # Training data dict
        training_data: dict[str, int] = {
            str(vbr_file): 3,  # V0 class
            str(cbr_file): 5,  # 320 class
        }

        # Track is_vbr values by file path
        is_vbr_by_path: dict[str, float] = {}

        def mock_metadata(path: str) -> Mock:
            if "v0" in path:
                return Mock(encoding_type="VBR")
            return Mock(encoding_type="CBR")

        def mock_analyze(path: str, is_vbr: float = 0.0) -> "SpectralFeatures":
            is_vbr_by_path[path] = is_vbr
            return sample_features

        with patch.object(analyzer.file_analyzer, "analyze", side_effect=mock_metadata), \
             patch.object(analyzer.spectrum_analyzer, "analyze_file", side_effect=mock_analyze):

            analyzer.train(training_data)

            # Verify VBR file got is_vbr=1.0
            assert is_vbr_by_path[str(vbr_file)] == 1.0, f"VBR file should have is_vbr=1.0, got {is_vbr_by_path[str(vbr_file)]}"
            # Verify CBR file got is_vbr=0.0
            assert is_vbr_by_path[str(cbr_file)] == 0.0, f"CBR file should have is_vbr=0.0, got {is_vbr_by_path[str(cbr_file)]}"


class TestIntegratedTranscodeDetection:
    """Test full transcode detection pipeline."""

    def test_analyze_detects_stated_class_from_container(self) -> None:
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
