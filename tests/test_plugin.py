"""Tests for the beets plugin module.

Since BitraterPlugin inherits from beets.plugins.BeetsPlugin which isn't
available in the test environment, we test the plugin methods by importing
the source module and extracting the original class code before the beets
metaclass interferes.
"""

import importlib
import logging
from unittest.mock import MagicMock

import pytest

from bitrater.types import AnalysisResult

# We need beets available as imports. If beets isn't installed, mock it.
_beets_available = importlib.util.find_spec("beets") is not None

if not _beets_available:
    pytest.skip("beets not installed", allow_module_level=True)

from beets.ui import UserError  # noqa: E402

from beetsplug.bitrater.plugin import BitraterPlugin  # noqa: E402


@pytest.fixture(autouse=True)
def _propagate_beets_logger():
    """Enable log propagation so caplog can capture beets.bitrater output.

    The beets library sets propagate=False on the 'beets' logger and adds its
    own StreamHandler. This blocks caplog from seeing records. We temporarily
    re-enable propagation for test assertions.
    """
    beets_logger = logging.getLogger("beets")
    orig_propagate = beets_logger.propagate
    beets_logger.propagate = True
    yield
    beets_logger.propagate = orig_propagate


@pytest.fixture
def plugin():
    """Create a BitraterPlugin instance with mocked internals.

    NOTE: Uses object.__new__() to bypass __init__() because BitraterPlugin's
    __init__ calls super().__init__() which requires the full beets config
    system (confuse, beets.config, plugin template registration). This means:
    - Plugin initialization logic (model loading, event registration, config
      defaults) is NOT tested by this fixture.
    - Any new required attributes added in __init__ must be manually added here.
    """
    p = object.__new__(BitraterPlugin)
    p.analyzer = MagicMock()
    p.config = MagicMock()
    p.item_types = {}

    defaults = {
        "min_confidence": 0.7,
        "warn_transcodes": True,
        "threads": None,
        "auto": False,
        "model_path": None,
        "training_dir": None,
    }

    def make_config_item(key):
        item = MagicMock()
        item.get.return_value = defaults.get(key)
        return item

    p.config = {key: make_config_item(key) for key in defaults}
    return p


class TestCommands:
    """Tests for commands()."""

    def test_returns_bitrater_command(self, plugin):
        cmds = plugin.commands()
        assert len(cmds) == 1
        assert cmds[0].name == "bitrater"


class TestProcessResults:
    """Tests for _process_results."""

    def test_stores_results_on_items(self, plugin):
        item = MagicMock()
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
        )

        plugin._process_results([item], [result], verbose=False)

        assert item.original_bitrate == 320
        assert item.original_format == "320"
        assert item.bitrate_confidence == 0.95
        assert item.is_transcoded is False
        item.store.assert_called_once()

    def test_skips_none_results(self, plugin):
        item = MagicMock()
        plugin._process_results([item], [None], verbose=False)
        item.store.assert_not_called()

    def test_transcode_detected_warns(self, plugin):
        item = MagicMock()
        item.path = "/music/test.mp3"
        result = AnalysisResult(
            filename="test.mp3",
            file_format="mp3",
            original_format="128",
            original_bitrate=128,
            confidence=0.9,
            is_transcode=True,
            stated_class="320",
            detected_cutoff=16000,
            quality_gap=5,
            transcoded_from="128",
        )

        plugin._process_results([item], [result], verbose=False)
        assert item.is_transcoded is True

    def test_verbose_prints_analysis(self, plugin):
        item = MagicMock()
        item.title = "Test Song"
        item.path = "/music/test.mp3"
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
        )

        # Should not raise
        plugin._process_results([item], [result], verbose=True)


class TestPrintSummary:
    """Tests for _print_summary."""

    def test_all_ok(self, plugin, caplog):
        with caplog.at_level(logging.INFO, logger="beets"):
            plugin._print_summary(total=10, transcodes=0, low_confidence=0)
        assert "Total files analyzed: 10" in caplog.text
        assert "All files appear to be original quality" in caplog.text

    def test_with_transcodes(self, plugin, caplog):
        with caplog.at_level(logging.INFO, logger="beets"):
            plugin._print_summary(total=10, transcodes=3, low_confidence=0)
        assert "Potential transcodes detected: 3" in caplog.text

    def test_with_low_confidence(self, plugin, caplog):
        with caplog.at_level(logging.INFO, logger="beets"):
            plugin._print_summary(total=10, transcodes=0, low_confidence=2)
        assert "Low confidence results: 2" in caplog.text


class TestPrintAnalysis:
    """Tests for _print_analysis."""

    def test_prints_without_transcode(self, plugin, caplog):
        item = MagicMock()
        item.title = "Song"
        item.path = "/test.mp3"

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

        with caplog.at_level(logging.INFO, logger="beets"):
            plugin._print_analysis(item, result)
        assert "Song" in caplog.text
        assert "320" in caplog.text
        assert "90.0%" in caplog.text
        assert "TRANSCODE" not in caplog.text

    def test_prints_with_transcode_and_warnings(self, plugin, caplog):
        item = MagicMock()
        item.title = "Bad Song"
        item.path = "/test.mp3"

        result = AnalysisResult(
            filename="test.mp3",
            file_format="mp3",
            original_format="128",
            original_bitrate=128,
            confidence=0.6,
            is_transcode=True,
            stated_class="320",
            detected_cutoff=16000,
            quality_gap=5,
            transcoded_from="128",
            warnings=["Low confidence", "Possible transcode"],
        )

        with caplog.at_level(logging.INFO, logger="beets"):
            plugin._print_analysis(item, result)
        assert "Bad Song" in caplog.text
        assert "TRANSCODE DETECTED" in caplog.text
        assert "Low confidence" in caplog.text
        assert "Possible transcode" in caplog.text


class TestPrintValidationResults:
    """Tests for _print_validation_results."""

    def test_high_accuracy(self, plugin, caplog):
        metrics = {
            "total_samples": 100,
            "train_samples": 80,
            "test_samples": 20,
            "train_pct": 0.8,
            "test_pct": 0.2,
            "accuracy": 0.97,
            "per_class": {
                "128": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5},
            },
            "confusion_matrix": [[5]],
            "class_names": ["128"],
        }
        with caplog.at_level(logging.INFO, logger="beets"):
            plugin._print_validation_results(metrics)
        assert "MODEL VALIDATION RESULTS" in caplog.text
        assert "97.0%" in caplog.text
        assert "Matches or exceeds" in caplog.text

    def test_medium_accuracy(self, plugin, caplog):
        metrics = {
            "total_samples": 100,
            "train_samples": 80,
            "test_samples": 20,
            "train_pct": 0.8,
            "test_pct": 0.2,
            "accuracy": 0.92,
            "per_class": {},
            "confusion_matrix": [],
            "class_names": [],
        }
        with caplog.at_level(logging.INFO, logger="beets"):
            plugin._print_validation_results(metrics)
        assert "92.0%" in caplog.text
        assert "Slightly below" in caplog.text

    def test_low_accuracy(self, plugin, caplog):
        metrics = {
            "total_samples": 100,
            "train_samples": 80,
            "test_samples": 20,
            "train_pct": 0.8,
            "test_pct": 0.2,
            "accuracy": 0.74,
            "per_class": {},
            "confusion_matrix": [],
            "class_names": [],
        }
        with caplog.at_level(logging.INFO, logger="beets"):
            plugin._print_validation_results(metrics)
        assert "74.0%" in caplog.text
        assert "Below expected accuracy" in caplog.text


class TestTrainClassifier:
    """Tests for _train_classifier."""

    def test_no_training_dir_raises(self, plugin):
        opts = MagicMock()
        opts.training_dir = None
        opts.verbose = False
        plugin.config["training_dir"].get.return_value = None

        with pytest.raises(UserError):
            plugin._train_classifier(opts)

    def test_missing_training_dir_raises(self, plugin, tmp_path):
        opts = MagicMock()
        opts.training_dir = str(tmp_path / "nonexistent")
        opts.verbose = False

        with pytest.raises(UserError):
            plugin._train_classifier(opts)

    def test_train_success(self, plugin, tmp_path):
        training_dir = tmp_path / "training"
        training_dir.mkdir()

        opts = MagicMock()
        opts.training_dir = str(training_dir)
        opts.verbose = False
        opts.validate = False
        opts.save_model = None
        opts.threads = None

        plugin.analyzer.train_from_directory.return_value = {
            "successful": 100,
            "total_files": 100,
        }

        plugin._train_classifier(opts)
        plugin.analyzer.train_from_directory.assert_called_once()

    def test_train_with_save_model(self, plugin, tmp_path):
        training_dir = tmp_path / "training"
        training_dir.mkdir()

        opts = MagicMock()
        opts.training_dir = str(training_dir)
        opts.verbose = False
        opts.validate = False
        opts.save_model = str(tmp_path / "model.pkl")
        opts.threads = 4

        plugin.analyzer.train_from_directory.return_value = {
            "successful": 100,
            "total_files": 100,
        }

        plugin._train_classifier(opts)
        plugin.analyzer.save_model.assert_called_once()

    def test_validate_mode(self, plugin, tmp_path):
        training_dir = tmp_path / "training"
        training_dir.mkdir()

        opts = MagicMock()
        opts.training_dir = str(training_dir)
        opts.verbose = False
        opts.validate = True
        opts.threads = None

        plugin.analyzer.validate_from_directory.return_value = {
            "total_samples": 100,
            "train_samples": 80,
            "test_samples": 20,
            "train_pct": 0.8,
            "test_pct": 0.2,
            "accuracy": 0.85,
            "per_class": {},
            "confusion_matrix": [],
            "class_names": [],
        }

        plugin._train_classifier(opts)
        plugin.analyzer.validate_from_directory.assert_called_once()

    def test_verbose_enables_logging(self, plugin, tmp_path):
        training_dir = tmp_path / "training"
        training_dir.mkdir()

        opts = MagicMock()
        opts.training_dir = str(training_dir)
        opts.verbose = True
        opts.validate = False
        opts.save_model = None
        opts.threads = None

        plugin.analyzer.train_from_directory.return_value = {
            "successful": 100,
            "total_files": 100,
        }

        plugin._train_classifier(opts)


class TestAnalyzeCommand:
    """Tests for analyze_command."""

    def test_no_items(self, plugin):
        lib = MagicMock()
        lib.items.return_value = []

        opts = MagicMock()
        opts.train = False
        opts.verbose = False
        opts.threads = None

        plugin.analyzer.is_trained = True
        plugin.analyze_command(lib, opts, [])

    def test_with_train_flag(self, plugin, tmp_path):
        training_dir = tmp_path / "training"
        training_dir.mkdir()

        lib = MagicMock()
        opts = MagicMock()
        opts.train = True
        opts.training_dir = str(training_dir)
        opts.verbose = False
        opts.validate = False
        opts.save_model = None
        opts.threads = None

        plugin.analyzer.train_from_directory.return_value = {
            "successful": 100,
            "total_files": 100,
        }

        plugin.analyze_command(lib, opts, [])
        plugin.analyzer.train_from_directory.assert_called_once()


class TestImportTask:
    """Tests for import_task."""

    def test_auto_disabled(self, plugin):
        """When auto is False, import_task should do nothing."""
        plugin.config["auto"].get.return_value = False
        session = MagicMock()
        task = MagicMock()

        plugin.import_task(session, task)

    def test_model_not_trained(self, plugin):
        """When model is not trained, import_task should skip."""
        plugin.config["auto"].get.return_value = True
        plugin.analyzer.is_trained = False

        session = MagicMock()
        task = MagicMock()

        plugin.import_task(session, task)

    def test_auto_analysis_runs(self, plugin):
        """When auto is True and model trained, analyze imported files."""
        plugin.config["auto"].get.return_value = True
        plugin.config["warn_transcodes"].get.return_value = True
        plugin.analyzer.is_trained = True

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
        )
        plugin.analyzer.analyze_file.return_value = result

        item = MagicMock()
        item.path = "/test.mp3"

        session = MagicMock()
        task = MagicMock()
        task.items = [item]

        plugin.import_task(session, task)

        plugin.analyzer.analyze_file.assert_called_once()
        assert item.original_bitrate == 320

    def test_auto_analysis_error_handled(self, plugin):
        """Errors during auto-analysis should be caught."""
        plugin.config["auto"].get.return_value = True
        plugin.analyzer.is_trained = True
        plugin.analyzer.analyze_file.side_effect = RuntimeError("analysis error")

        item = MagicMock()
        item.path = "/test.mp3"

        session = MagicMock()
        task = MagicMock()
        task.items = [item]

        # Should not raise
        plugin.import_task(session, task)


class TestEnableVerboseLogging:
    """Tests for _enable_verbose_logging."""

    def test_sets_log_level(self, plugin):
        plugin._enable_verbose_logging()
        bitrater_logger = logging.getLogger("beets.bitrater")
        assert bitrater_logger.level == logging.INFO
        assert bitrater_logger.propagate is True
