"""Tests for audio quality analyzer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from bitrater.analyzer import AudioQualityAnalyzer
from bitrater.cutoff_detector import CutoffResult
from bitrater.types import AnalysisResult, ClassifierPrediction, SpectralFeatures


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

    def test_analyze_file_untrained_returns_unknown(
        self, sample_features: SpectralFeatures, tmp_path: Path
    ) -> None:
        """Test analyze_file returns UNKNOWN when classifier not trained."""
        analyzer = AudioQualityAnalyzer()

        # Create a real temporary file to avoid mocking Path.exists
        fake_audio = tmp_path / "test.mp3"
        fake_audio.write_bytes(b"fake audio content")

        # Only mock the components that do actual audio processing
        with (
            patch.object(analyzer.spectrum_analyzer, "analyze_file", return_value=sample_features),
            patch.object(analyzer.file_analyzer, "analyze", return_value=Mock(bitrate=320)),
        ):

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
    """Tests for warning generation through the real analyze_file pipeline."""

    def _make_analyzer(
        self,
        sample_features: SpectralFeatures,
        tmp_path: Path,
        *,
        file_ext: str = "mp3",
        metadata_bitrate: int | None = 320,
        metadata_encoding: str = "CBR",
        classifier_format: str = "320",
        classifier_bitrate: int = 320,
        classifier_confidence: float = 0.95,
        cutoff_frequency: int = 20500,
        cutoff_gradient: float = 0.8,
    ) -> tuple[AudioQualityAnalyzer, str]:
        """Create an analyzer with controlled mocks, letting real warning logic run."""
        analyzer = AudioQualityAnalyzer()

        fake_audio = tmp_path / f"test.{file_ext}"
        fake_audio.write_bytes(b"fake audio content")

        # Mock metadata
        if metadata_bitrate is not None:
            mock_metadata = Mock(bitrate=metadata_bitrate, encoding_type=metadata_encoding)
        else:
            mock_metadata = Mock(bitrate=None, encoding_type=metadata_encoding)

        # Mock classifier prediction
        mock_prediction = ClassifierPrediction(
            format_type=classifier_format,
            estimated_bitrate=classifier_bitrate,
            confidence=classifier_confidence,
        )

        # Mock cutoff detection
        psd = np.ones(100, dtype=np.float32)
        freqs = np.linspace(15000, 22050, 100)
        mock_cutoff = CutoffResult(
            cutoff_frequency=cutoff_frequency,
            gradient=cutoff_gradient,
            is_sharp=cutoff_gradient > 0.5,
            confidence=0.8,
        )

        # Apply mocks — let confidence_calculator, transcode_detector, and
        # warning assembly in analyze_file() run for real
        patch.object(analyzer.file_analyzer, "analyze", return_value=mock_metadata).start()
        patch.object(
            analyzer.spectrum_analyzer, "analyze_file", return_value=sample_features
        ).start()
        patch.object(analyzer.classifier, "predict", return_value=mock_prediction).start()
        analyzer.classifier.trained = True
        patch.object(analyzer.spectrum_analyzer, "get_psd", return_value=(psd, freqs)).start()
        patch.object(analyzer.cutoff_detector, "detect", return_value=mock_cutoff).start()

        return analyzer, str(fake_audio)

    def test_low_confidence_warning(
        self, sample_features: SpectralFeatures, tmp_path: Path
    ) -> None:
        """Low classifier confidence + cutoff mismatch + soft gradient → confidence warning."""
        analyzer, path = self._make_analyzer(
            sample_features,
            tmp_path,
            classifier_confidence=0.4,
            classifier_format="320",
            classifier_bitrate=320,
            cutoff_frequency=16000,  # Big mismatch from expected ~20500
            cutoff_gradient=0.1,  # Soft gradient → penalty
        )
        result = analyzer.analyze_file(path)

        assert result is not None
        assert any("confidence" in w.lower() for w in result.warnings)

    def test_transcode_warning(self, sample_features: SpectralFeatures, tmp_path: Path) -> None:
        """FLAC container + classifier detects 128 → transcode warning."""
        analyzer, path = self._make_analyzer(
            sample_features,
            tmp_path,
            file_ext="flac",
            metadata_bitrate=None,
            metadata_encoding="lossless",
            classifier_format="128",
            classifier_bitrate=128,
            cutoff_frequency=16000,
            cutoff_gradient=0.9,
        )
        result = analyzer.analyze_file(path)

        assert result is not None
        assert result.is_transcode is True
        assert any("transcode" in w.lower() for w in result.warnings)

    def test_bitrate_mismatch_warning(
        self, sample_features: SpectralFeatures, tmp_path: Path
    ) -> None:
        """Stated 320 kbps but classifier says 128 → bitrate mismatch warning."""
        analyzer, path = self._make_analyzer(
            sample_features,
            tmp_path,
            metadata_bitrate=320,
            classifier_format="128",
            classifier_bitrate=128,
            cutoff_frequency=16000,
            cutoff_gradient=0.9,
        )
        result = analyzer.analyze_file(path)

        assert result is not None
        assert any("bitrate" in w.lower() or "upsampled" in w.lower() for w in result.warnings)


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


class TestIntegratedTranscodeDetection:
    """Test full transcode detection pipeline."""

    def test_analyze_detects_stated_class_from_container(self) -> None:
        """Analyzer should determine stated_class from file format."""
        # This test will need mock spectral data
        # For now, test the helper method
        from bitrater.analyzer import AudioQualityAnalyzer

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
