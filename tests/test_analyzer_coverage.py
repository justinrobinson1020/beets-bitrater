"""Additional tests for analyzer.py to improve coverage."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bitrater.analyzer import AudioQualityAnalyzer, _extract_features_worker
from bitrater.types import FileMetadata, SpectralFeatures


class TestGetStatedClass:
    """Tests for _get_stated_class."""

    def setup_method(self):
        self.analyzer = AudioQualityAnalyzer()

    def test_flac_is_lossless(self):
        assert self.analyzer._get_stated_class("flac", None) == "LOSSLESS"

    def test_wav_is_lossless(self):
        assert self.analyzer._get_stated_class("wav", None) == "LOSSLESS"

    def test_alac_is_lossless(self):
        assert self.analyzer._get_stated_class("alac", None) == "LOSSLESS"

    def test_mp3_no_bitrate_is_unknown(self):
        assert self.analyzer._get_stated_class("mp3", None) == "UNKNOWN"

    def test_mp3_128(self):
        assert self.analyzer._get_stated_class("mp3", 128) == "128"

    def test_mp3_140_is_128(self):
        assert self.analyzer._get_stated_class("mp3", 140) == "128"

    def test_mp3_170_is_v2(self):
        assert self.analyzer._get_stated_class("mp3", 170) == "V2"

    def test_mp3_192(self):
        assert self.analyzer._get_stated_class("mp3", 192) == "192"

    def test_mp3_245_is_v0(self):
        assert self.analyzer._get_stated_class("mp3", 245) == "V0"

    def test_mp3_260_is_v0(self):
        # 256 and 260 fall in V0 range (210-260)
        assert self.analyzer._get_stated_class("mp3", 260) == "V0"

    def test_mp3_270_is_256(self):
        assert self.analyzer._get_stated_class("mp3", 270) == "256"

    def test_mp3_290_is_256(self):
        assert self.analyzer._get_stated_class("mp3", 290) == "256"

    def test_mp3_320(self):
        assert self.analyzer._get_stated_class("mp3", 320) == "320"


class TestAnalyzeFileWithTrainedModel:
    """Tests for analyze_file when the classifier is trained."""

    def setup_method(self):
        self.analyzer = AudioQualityAnalyzer()

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_full_pipeline(self):
        analyzer = AudioQualityAnalyzer()

        # Mock all components
        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = FileMetadata(
            format="mp3",
            sample_rate=44100,
            duration=180.0,
            channels=2,
            encoding_type="CBR",
            encoder="LAME",
            bitrate=320,
        )

        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer = MagicMock()
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features
        analyzer.spectrum_analyzer.get_psd.return_value = (
            np.zeros(100),
            np.linspace(0, 22050, 100),
        )

        analyzer.classifier = MagicMock()
        analyzer.classifier.trained = True
        analyzer.classifier.predict.return_value = MagicMock(
            format_type="320",
            estimated_bitrate=320,
            confidence=0.95,
        )

        analyzer.cutoff_detector = MagicMock()
        analyzer.cutoff_detector.detect.return_value = MagicMock(
            cutoff_frequency=20500,
            gradient=0.8,
        )

        analyzer.confidence_calculator = MagicMock()
        analyzer.confidence_calculator.calculate.return_value = MagicMock(
            final_confidence=0.92,
            warnings=[],
        )

        analyzer.transcode_detector = MagicMock()
        analyzer.transcode_detector.detect.return_value = MagicMock(
            is_transcode=False,
            quality_gap=0,
            transcoded_from=None,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is not None
        assert result.original_format == "320"
        assert result.confidence == 0.92
        assert result.is_transcode is False

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_no_psd_data(self):
        analyzer = AudioQualityAnalyzer()

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = None

        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer = MagicMock()
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features
        analyzer.spectrum_analyzer.get_psd.return_value = None  # No PSD data

        analyzer.classifier = MagicMock()
        analyzer.classifier.trained = True
        analyzer.classifier.predict.return_value = MagicMock(
            format_type="320",
            estimated_bitrate=320,
            confidence=0.9,
        )

        analyzer.confidence_calculator = MagicMock()
        analyzer.confidence_calculator.calculate.return_value = MagicMock(
            final_confidence=0.85,
            warnings=[],
        )

        analyzer.transcode_detector = MagicMock()
        analyzer.transcode_detector.detect.return_value = MagicMock(
            is_transcode=False,
            quality_gap=0,
            transcoded_from=None,
        )

        analyzer.cutoff_detector = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is not None
        assert result.detected_cutoff == 0

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_metadata_error(self):
        """Metadata extraction failure should not prevent analysis."""
        analyzer = AudioQualityAnalyzer()

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.side_effect = ValueError("bad format")

        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer = MagicMock()
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features
        analyzer.spectrum_analyzer.get_psd.return_value = None

        analyzer.classifier = MagicMock()
        analyzer.classifier.trained = True
        analyzer.classifier.predict.return_value = MagicMock(
            format_type="192",
            estimated_bitrate=192,
            confidence=0.8,
        )

        analyzer.confidence_calculator = MagicMock()
        analyzer.confidence_calculator.calculate.return_value = MagicMock(
            final_confidence=0.75,
            warnings=[],
        )

        analyzer.transcode_detector = MagicMock()
        analyzer.transcode_detector.detect.return_value = MagicMock(
            is_transcode=False,
            quality_gap=0,
            transcoded_from=None,
        )

        analyzer.cutoff_detector = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is not None

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_unexpected_metadata_error(self):
        """Unexpected metadata errors should be logged but not prevent analysis."""
        analyzer = AudioQualityAnalyzer()

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.side_effect = TypeError("unexpected")

        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer = MagicMock()
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features
        analyzer.spectrum_analyzer.get_psd.return_value = None

        analyzer.classifier = MagicMock()
        analyzer.classifier.trained = True
        analyzer.classifier.predict.return_value = MagicMock(
            format_type="320",
            estimated_bitrate=320,
            confidence=0.9,
        )

        analyzer.confidence_calculator = MagicMock()
        analyzer.confidence_calculator.calculate.return_value = MagicMock(
            final_confidence=0.85,
            warnings=[],
        )

        analyzer.transcode_detector = MagicMock()
        analyzer.transcode_detector.detect.return_value = MagicMock(
            is_transcode=False,
            quality_gap=0,
            transcoded_from=None,
        )

        analyzer.cutoff_detector = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is not None

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_spectral_failure(self):
        """Spectral analysis failure returns None."""
        analyzer = AudioQualityAnalyzer()

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = None

        analyzer.spectrum_analyzer = MagicMock()
        analyzer.spectrum_analyzer.analyze_file.return_value = None

        analyzer.classifier = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is None

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_bitrate_mismatch_warning(self):
        """Bitrate mismatch should generate a warning."""
        analyzer = AudioQualityAnalyzer()

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = FileMetadata(
            format="mp3",
            sample_rate=44100,
            duration=180.0,
            channels=2,
            encoding_type="CBR",
            encoder="LAME",
            bitrate=320,  # Claims 320
        )

        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer = MagicMock()
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features
        analyzer.spectrum_analyzer.get_psd.return_value = None

        analyzer.classifier = MagicMock()
        analyzer.classifier.trained = True
        analyzer.classifier.predict.return_value = MagicMock(
            format_type="128",  # Detected as 128
            estimated_bitrate=128,
            confidence=0.9,
        )

        analyzer.confidence_calculator = MagicMock()
        analyzer.confidence_calculator.calculate.return_value = MagicMock(
            final_confidence=0.85,
            warnings=[],
        )

        analyzer.transcode_detector = MagicMock()
        analyzer.transcode_detector.detect.return_value = MagicMock(
            is_transcode=True,
            quality_gap=5,
            transcoded_from="128",
        )

        analyzer.cutoff_detector = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is not None
        assert any("bitrate" in w.lower() or "upsampled" in w.lower() for w in result.warnings)
        assert any("transcoded" in w.lower() for w in result.warnings)


class TestCollectTrainingDataNested:
    """Tests for _collect_training_data with nested structure."""

    def test_nested_structure(self, tmp_path):
        # Create nested structure
        encoded = tmp_path / "encoded" / "lossy"
        for d in ["128", "192", "256", "320", "v0", "v2"]:
            (encoded / d).mkdir(parents=True)
            (encoded / d / "test.mp3").touch()

        lossless = tmp_path / "lossless"
        lossless.mkdir()
        (lossless / "test.flac").touch()

        analyzer = AudioQualityAnalyzer()
        data = analyzer._collect_training_data(tmp_path)

        assert len(data) == 7  # 6 lossy + 1 lossless

    def test_missing_lossless_dir_warns(self, tmp_path):
        # Only create lossy
        lossy = tmp_path / "128"
        lossy.mkdir()
        (lossy / "test.mp3").touch()

        analyzer = AudioQualityAnalyzer()
        data = analyzer._collect_training_data(tmp_path)
        assert len(data) == 1


class TestSaveLoadModel:
    """Tests for save_model and load_model."""

    def test_save_model_delegates_to_classifier(self):
        analyzer = AudioQualityAnalyzer()
        analyzer.classifier = MagicMock()
        analyzer.save_model(Path("/tmp/model.pkl"))
        analyzer.classifier.save_model.assert_called_once_with(Path("/tmp/model.pkl"))

    def test_load_model_delegates_to_classifier(self):
        analyzer = AudioQualityAnalyzer()
        analyzer.classifier = MagicMock()
        analyzer.load_model(Path("/tmp/model.pkl"))
        analyzer.classifier.load_model.assert_called_once_with(Path("/tmp/model.pkl"))


class TestTrainSequential:
    """Tests for the sequential train method."""

    def test_train_empty_raises(self):
        analyzer = AudioQualityAnalyzer()
        with pytest.raises(ValueError, match="No training data"):
            analyzer.train({})

    def test_train_with_mock_features(self):
        analyzer = AudioQualityAnalyzer()

        # Mock spectrum analyzer to return features
        num_bands = 150
        mock_features = SpectralFeatures(
            features=np.random.rand(num_bands).astype(np.float32),
            frequency_bands=[(16000 + i * 40, 16040 + i * 40) for i in range(num_bands)],
        )
        analyzer.spectrum_analyzer = MagicMock()
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features
        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = None

        # Mock classifier
        analyzer.classifier = MagicMock()

        # Create fake training data with enough samples
        training_data = {}
        files = []
        for i in range(14):  # 2 per class
            f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            f.close()
            files.append(f.name)
            training_data[f.name] = i % 7

        try:
            stats = analyzer.train(training_data)
            assert stats["total_files"] == 14
            assert stats["successful"] == 14
            analyzer.classifier.train.assert_called_once()
        finally:
            for f in files:
                os.unlink(f)


class TestTrainFromDirectory:
    """Tests for train_from_directory."""

    def test_sequential_when_workers_1(self, tmp_path):
        # Create minimal training dir
        d = tmp_path / "128"
        d.mkdir()
        (d / "test.mp3").touch()

        analyzer = AudioQualityAnalyzer()
        analyzer.train = MagicMock(return_value={"total_files": 1, "successful": 1, "failed": 0})

        analyzer.train_from_directory(tmp_path, num_workers=1)
        analyzer.train.assert_called_once()

    def test_parallel_by_default(self, tmp_path):
        d = tmp_path / "128"
        d.mkdir()
        (d / "test.mp3").touch()

        analyzer = AudioQualityAnalyzer()
        analyzer.train_parallel = MagicMock(
            return_value={"total_files": 1, "successful": 1, "failed": 0}
        )

        analyzer.train_from_directory(tmp_path)
        analyzer.train_parallel.assert_called_once()

    def test_explicit_use_parallel_false(self, tmp_path):
        d = tmp_path / "128"
        d.mkdir()
        (d / "test.mp3").touch()

        analyzer = AudioQualityAnalyzer()
        analyzer.train = MagicMock(return_value={"total_files": 1, "successful": 1, "failed": 0})

        analyzer.train_from_directory(tmp_path, use_parallel=False)
        analyzer.train.assert_called_once()


class TestAnalyzeFileWarningLogic:
    """Tests for analyze_file's warning generation logic.

    Uses real ConfidenceCalculator, TranscodeDetector, and CutoffDetector
    instances. Only mocks I/O-bound components (file_analyzer, spectrum_analyzer,
    classifier) to verify that the orchestration logic in analyze_file correctly
    generates warnings based on real component decisions.
    """

    def _make_analyzer(self):
        """Create an analyzer with real decision components, mocked I/O."""
        from bitrater.confidence import ConfidenceCalculator
        from bitrater.cutoff_detector import CutoffDetector
        from bitrater.transcode_detector import TranscodeDetector

        analyzer = object.__new__(AudioQualityAnalyzer)
        analyzer.file_analyzer = MagicMock()
        analyzer.spectrum_analyzer = MagicMock()
        analyzer.classifier = MagicMock()
        analyzer.classifier.trained = True

        # Real decision components
        analyzer.cutoff_detector = CutoffDetector()
        analyzer.confidence_calculator = ConfidenceCalculator()
        analyzer.transcode_detector = TranscodeDetector()
        return analyzer

    def test_low_confidence_generates_warning(self, tmp_path):
        """Confidence below LOW_CONFIDENCE_THRESHOLD (0.7) should produce a warning."""
        analyzer = self._make_analyzer()

        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        analyzer.file_analyzer.analyze.return_value = FileMetadata(
            format="mp3",
            sample_rate=44100,
            duration=180.0,
            channels=2,
            encoding_type="CBR",
            encoder="LAME",
            bitrate=320,
        )
        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features

        # Classifier returns low confidence
        analyzer.classifier.predict.return_value = MagicMock(
            format_type="320",
            estimated_bitrate=320,
            confidence=0.3,
        )
        # No PSD data → neutral gradient (0.5)
        analyzer.spectrum_analyzer.get_psd.return_value = None

        result = analyzer.analyze_file(str(audio_file))

        assert result is not None
        assert result.confidence < 0.7
        assert any("Low confidence" in w for w in result.warnings)

    def test_transcode_detection_generates_warning(self, tmp_path):
        """File claiming 320 but detected as 128 should produce transcode warning."""
        analyzer = self._make_analyzer()

        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        analyzer.file_analyzer.analyze.return_value = FileMetadata(
            format="mp3",
            sample_rate=44100,
            duration=180.0,
            channels=2,
            encoding_type="CBR",
            encoder="LAME",
            bitrate=320,
        )
        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features

        # Classifier says 128 but file claims 320
        analyzer.classifier.predict.return_value = MagicMock(
            format_type="128",
            estimated_bitrate=128,
            confidence=0.95,
        )
        analyzer.spectrum_analyzer.get_psd.return_value = None

        result = analyzer.analyze_file(str(audio_file))

        assert result is not None
        assert result.is_transcode is True
        assert result.transcoded_from == "128"
        assert result.quality_gap > 0
        assert any("transcoded from 128" in w for w in result.warnings)

    def test_bitrate_mismatch_generates_upsampled_warning(self, tmp_path):
        """Stated bitrate > detected * 1.5 should produce upsampled warning."""
        analyzer = self._make_analyzer()

        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        # File metadata claims 320 kbps
        analyzer.file_analyzer.analyze.return_value = FileMetadata(
            format="mp3",
            sample_rate=44100,
            duration=180.0,
            channels=2,
            encoding_type="CBR",
            encoder="LAME",
            bitrate=320,
        )
        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features

        # Classifier detects 192 (320 > 192 * 1.5 = 288, so mismatch triggers)
        analyzer.classifier.predict.return_value = MagicMock(
            format_type="192",
            estimated_bitrate=192,
            confidence=0.95,
        )
        analyzer.spectrum_analyzer.get_psd.return_value = None

        result = analyzer.analyze_file(str(audio_file))

        assert result is not None
        assert any("upsampled" in w.lower() for w in result.warnings)

    def test_no_bitrate_mismatch_when_within_factor(self, tmp_path):
        """Stated bitrate close to detected should NOT produce upsampled warning."""
        analyzer = self._make_analyzer()

        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        # File metadata claims 320 kbps
        analyzer.file_analyzer.analyze.return_value = FileMetadata(
            format="mp3",
            sample_rate=44100,
            duration=180.0,
            channels=2,
            encoding_type="CBR",
            encoder="LAME",
            bitrate=320,
        )
        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features

        # Classifier also detects 320 (no mismatch)
        analyzer.classifier.predict.return_value = MagicMock(
            format_type="320",
            estimated_bitrate=320,
            confidence=0.95,
        )
        analyzer.spectrum_analyzer.get_psd.return_value = None

        result = analyzer.analyze_file(str(audio_file))

        assert result is not None
        assert not any("upsampled" in w.lower() for w in result.warnings)

    def test_lossless_detected_skips_bitrate_mismatch(self, tmp_path):
        """Lossless detection should not trigger bitrate mismatch warning."""
        analyzer = self._make_analyzer()

        audio_file = tmp_path / "test.flac"
        audio_file.touch()

        analyzer.file_analyzer.analyze.return_value = FileMetadata(
            format="flac",
            sample_rate=44100,
            duration=180.0,
            channels=2,
            encoding_type="lossless",
            encoder="Unknown",
            bitrate=None,
        )
        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features

        analyzer.classifier.predict.return_value = MagicMock(
            format_type="LOSSLESS",
            estimated_bitrate=1411,
            confidence=0.95,
        )
        analyzer.spectrum_analyzer.get_psd.return_value = None

        result = analyzer.analyze_file(str(audio_file))

        assert result is not None
        assert not any("upsampled" in w.lower() for w in result.warnings)

    def test_untrained_classifier_returns_unknown(self, tmp_path):
        """Untrained classifier should return UNKNOWN with warning."""
        analyzer = self._make_analyzer()
        analyzer.classifier.trained = False

        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        analyzer.file_analyzer.analyze.return_value = None
        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features

        result = analyzer.analyze_file(str(audio_file))

        assert result is not None
        assert result.original_format == "UNKNOWN"
        assert result.confidence == 0.0
        assert "Classifier not trained" in result.warnings

    def test_cutoff_with_psd_data_applies_penalties(self, tmp_path):
        """When PSD data is available, cutoff detection should influence confidence."""
        analyzer = self._make_analyzer()

        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        analyzer.file_analyzer.analyze.return_value = FileMetadata(
            format="mp3",
            sample_rate=44100,
            duration=180.0,
            channels=2,
            encoding_type="CBR",
            encoder="LAME",
            bitrate=320,
        )
        mock_features = MagicMock(spec=SpectralFeatures)
        analyzer.spectrum_analyzer.analyze_file.return_value = mock_features

        analyzer.classifier.predict.return_value = MagicMock(
            format_type="320",
            estimated_bitrate=320,
            confidence=0.95,
        )

        # Provide real PSD data with content dropping off around 20kHz
        freqs = np.linspace(0, 22050, 4097)
        psd = np.ones(4097) * 1e-3
        psd[freqs < 20000] = 1.0  # Content below 20kHz
        analyzer.spectrum_analyzer.get_psd.return_value = (psd, freqs)

        result = analyzer.analyze_file(str(audio_file))

        assert result is not None
        assert result.detected_cutoff > 0
        # Confidence should reflect cutoff detection (may have penalties)
        assert 0.0 < result.confidence <= 1.0


class TestExtractFeaturesWorkerErrors:
    """Tests for _extract_features_worker error handling."""

    @patch("bitrater.analyzer._worker_analyzer")
    @patch("bitrater.analyzer._init_worker")
    def test_runtime_error_returns_none(self, mock_init, mock_sa):
        mock_sa.analyze_file.side_effect = RuntimeError("analysis failed")

        path, features = _extract_features_worker("/nonexistent/file.mp3")
        assert features is None

    @patch("bitrater.analyzer._worker_analyzer")
    @patch("bitrater.analyzer._init_worker")
    def test_unexpected_error_returns_none(self, mock_init, mock_sa):
        mock_sa.analyze_file.side_effect = Exception("unexpected")

        path, features = _extract_features_worker("/nonexistent/file.mp3")
        assert features is None
