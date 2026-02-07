"""Tests for model version checking on load."""

import logging
import pickle

import pytest

from beetsplug.bitrater.classifier import QualityClassifier


class TestModelVersionCheck:
    """Tests for model version check during load_model."""

    def _train_and_save(self, training_features, temp_model_path):
        """Helper to train and save a model."""
        classifier = QualityClassifier()
        features_list, labels = training_features
        classifier.train(features_list, labels, save_path=temp_model_path)
        return classifier

    def test_load_model_warns_on_version_mismatch(
        self, training_features, temp_model_path
    ):
        """load_model should log warning when model version doesn't match '3.0'."""
        self._train_and_save(training_features, temp_model_path)

        # Tamper with version in saved model
        with open(temp_model_path, "rb") as f:
            data = pickle.load(f)
        data["version"] = "1.0"
        with open(temp_model_path, "wb") as f:
            pickle.dump(data, f)

        # Load should warn about version mismatch
        classifier2 = QualityClassifier()
        logger = logging.getLogger("beets.bitrater")
        with _capture_handler(logger) as handler:
            classifier2.load_model(temp_model_path)

        assert any("version" in r.message.lower() for r in handler.records), (
            "Expected a warning about model version mismatch"
        )

    def test_load_model_no_warning_on_matching_version(
        self, training_features, temp_model_path
    ):
        """load_model should not warn when version matches '3.0'."""
        self._train_and_save(training_features, temp_model_path)

        classifier2 = QualityClassifier()
        logger = logging.getLogger("beets.bitrater")
        with _capture_handler(logger) as handler:
            classifier2.load_model(temp_model_path)

        assert not any(
            "version" in r.message.lower()
            for r in handler.records
            if r.levelno >= logging.WARNING
        ), "Should not warn when version matches"

    def test_load_model_warns_on_missing_version(
        self, training_features, temp_model_path
    ):
        """load_model should log warning when version key is absent."""
        self._train_and_save(training_features, temp_model_path)

        # Remove version key
        with open(temp_model_path, "rb") as f:
            data = pickle.load(f)
        del data["version"]
        with open(temp_model_path, "wb") as f:
            pickle.dump(data, f)

        classifier2 = QualityClassifier()
        logger = logging.getLogger("beets.bitrater")
        with _capture_handler(logger) as handler:
            classifier2.load_model(temp_model_path)

        assert any("version" in r.message.lower() for r in handler.records), (
            "Expected a warning about missing model version"
        )

    def test_load_model_logs_feature_count(
        self, training_features, temp_model_path
    ):
        """load_model should log the feature count from scaler."""
        self._train_and_save(training_features, temp_model_path)

        classifier2 = QualityClassifier()
        logger = logging.getLogger("beets.bitrater")
        with _capture_handler(logger) as handler:
            classifier2.load_model(temp_model_path)

        assert any("feature" in r.message.lower() for r in handler.records), (
            "Expected log message about feature count"
        )


class _CaptureHandler(logging.Handler):
    """Simple handler that captures log records."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


from contextlib import contextmanager


@contextmanager
def _capture_handler(logger):
    """Add a temporary capture handler to a logger."""
    handler = _CaptureHandler()
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
