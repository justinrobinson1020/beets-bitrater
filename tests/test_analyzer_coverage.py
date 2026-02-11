"""Additional tests for analyzer.py to improve coverage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bitrater.analyzer import AudioQualityAnalyzer, _extract_features_worker
from bitrater.types import AnalysisResult, FileMetadata, SpectralFeatures


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
        analyzer.spectrum_analyzer.get_psd.return_value = (np.zeros(100), np.linspace(0, 22050, 100))

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

        # Create a temp file
        import tempfile
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

        import tempfile
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

        import tempfile
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

        import tempfile
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

        import tempfile
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

        import tempfile
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
        import tempfile, os
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
        analyzer.train_parallel = MagicMock(return_value={"total_files": 1, "successful": 1, "failed": 0})

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


class TestExtractFeaturesWorkerErrors:
    """Tests for _extract_features_worker error handling."""

    @patch("bitrater.analyzer._worker_analyzer")
    @patch("bitrater.analyzer._worker_file_analyzer")
    @patch("bitrater.analyzer._init_worker")
    def test_runtime_error_returns_none(self, mock_init, mock_fa, mock_sa):
        mock_sa.analyze_file.side_effect = RuntimeError("analysis failed")
        mock_fa.analyze.return_value = None

        path, features = _extract_features_worker("/nonexistent/file.mp3")
        assert features is None

    @patch("bitrater.analyzer._worker_analyzer")
    @patch("bitrater.analyzer._worker_file_analyzer")
    @patch("bitrater.analyzer._init_worker")
    def test_unexpected_error_returns_none(self, mock_init, mock_fa, mock_sa):
        mock_sa.analyze_file.side_effect = Exception("unexpected")
        mock_fa.analyze.return_value = None

        path, features = _extract_features_worker("/nonexistent/file.mp3")
        assert features is None
