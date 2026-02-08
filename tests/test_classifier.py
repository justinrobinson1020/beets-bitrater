"""Tests for quality classifier."""

from __future__ import annotations

import numpy as np
import pytest

from beetsplug.bitrater.classifier import QualityClassifier
from beetsplug.bitrater.constants import BITRATE_CLASSES


class TestQualityClassifier:
    """Tests for QualityClassifier class."""

    def test_init(self) -> None:
        """Test classifier initialization with paper's SVM parameters."""
        classifier = QualityClassifier()

        assert classifier.trained is False
        assert classifier.classes == BITRATE_CLASSES

        # Verify SVM has correct parameters
        svm = classifier.classifier
        assert svm.kernel == "poly"
        assert svm.degree == 2  # Critical parameter from paper
        assert svm.gamma == 1
        assert svm.C == 1

    def test_extract_features(self, sample_features: SpectralFeatures) -> None:
        """Test feature extraction returns full feature vector."""
        classifier = QualityClassifier()
        features = classifier._extract_features(sample_features)

        # 150 PSD + 6 cutoff + 8 temporal + 6 artifact + 6 SFB21 + 4 rolloff + 1 is_vbr = 181
        assert features.shape == (181,)
        assert features.dtype == np.float32

    def test_extract_features_includes_is_vbr(self, sample_features: SpectralFeatures) -> None:
        """Test that extracted features include is_vbr at the end."""
        classifier = QualityClassifier()
        features = classifier._extract_features(sample_features)

        # Last feature should be the is_vbr value from SpectralFeatures
        assert features[-1] == sample_features.is_vbr

    def test_train(self, training_features: tuple[list, list], temp_model_path) -> None:
        """Test classifier training."""
        classifier = QualityClassifier()
        features_list, labels = training_features

        classifier.train(features_list, labels)

        assert classifier.trained is True

    def test_train_saves_model(self, training_features: tuple[list, list], temp_model_path) -> None:
        """Test that training can save model to file."""
        classifier = QualityClassifier()
        features_list, labels = training_features

        classifier.train(features_list, labels, save_path=temp_model_path)

        assert temp_model_path.exists()

    def test_train_empty_data_raises(self) -> None:
        """Test that training with empty data raises error."""
        classifier = QualityClassifier()

        with pytest.raises(ValueError):
            classifier.train([], [])

    def test_train_mismatched_lengths_raises(self, sample_features: SpectralFeatures) -> None:
        """Test that mismatched features/labels raises error."""
        classifier = QualityClassifier()

        with pytest.raises(ValueError):
            classifier.train([sample_features], [0, 1])

    def test_predict_untrained_raises(self, sample_features: SpectralFeatures) -> None:
        """Test that prediction without training raises error."""
        classifier = QualityClassifier()

        with pytest.raises(RuntimeError):
            classifier.predict(sample_features)

    def test_predict_returns_prediction(self, training_features: tuple[list, list], sample_features: SpectralFeatures) -> None:
        """Test that prediction returns ClassifierPrediction."""
        classifier = QualityClassifier()
        features_list, labels = training_features

        classifier.train(features_list, labels)
        prediction = classifier.predict(sample_features)

        assert prediction.format_type in ["128", "V2", "192", "V0", "256", "320", "LOSSLESS"]
        assert prediction.estimated_bitrate in [128, 190, 192, 245, 256, 320, 1411]
        assert 0 <= prediction.confidence <= 1
        assert isinstance(prediction.probabilities, dict)

    def test_save_and_load_model(self, training_features: tuple[list, list], temp_model_path, sample_features: SpectralFeatures) -> None:
        """Test model persistence."""
        # Train and save
        classifier1 = QualityClassifier()
        features_list, labels = training_features
        classifier1.train(features_list, labels, save_path=temp_model_path)

        # Get prediction from original
        pred1 = classifier1.predict(sample_features)

        # Load into new classifier
        classifier2 = QualityClassifier()
        classifier2.load_model(temp_model_path)

        assert classifier2.trained is True

        # Predictions should match
        pred2 = classifier2.predict(sample_features)
        assert pred1.format_type == pred2.format_type
        assert pred1.estimated_bitrate == pred2.estimated_bitrate

    def test_predict_batch(self, training_features: tuple[list, list], sample_features: SpectralFeatures, lossless_features: SpectralFeatures) -> None:
        """Test batch prediction."""
        classifier = QualityClassifier()
        features_list, labels = training_features
        classifier.train(features_list, labels)

        predictions = classifier.predict_batch([sample_features, lossless_features])

        assert len(predictions) == 2
        valid_formats = {fmt for fmt, _ in BITRATE_CLASSES.values()}
        assert all(p.format_type in valid_formats for p in predictions)


class TestBitrateClasses:
    """Tests for bitrate class configuration."""

    def test_seven_classes(self) -> None:
        """There should be exactly 7 quality classes."""
        assert len(BITRATE_CLASSES) == 7

    def test_class_indices(self) -> None:
        """Class indices should be 0-6."""
        assert set(BITRATE_CLASSES.keys()) == {0, 1, 2, 3, 4, 5, 6}

    def test_v2_class_exists(self) -> None:
        """V2 class should exist at index 1."""
        assert BITRATE_CLASSES[1] == ("V2", 190)
