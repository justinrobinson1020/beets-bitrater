"""Tests for quality classifier."""

from __future__ import annotations

import numpy as np
import pytest

from bitrater.classifier import QualityClassifier
from bitrater.constants import BITRATE_CLASSES
from bitrater.types import SpectralFeatures


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
        assert svm.gamma == 0.01
        assert svm.C == 100

    def test_extract_features(self, sample_features: SpectralFeatures) -> None:
        """Test feature extraction returns full feature vector."""
        classifier = QualityClassifier()
        features = classifier._extract_features(sample_features)

        # 150 PSD + 6 cutoff + 6 SFB21 + 4 rolloff + 6 discriminative + 1 is_vbr = 173
        assert features.shape == (173,)
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

    def test_predict_returns_prediction(
        self, training_features: tuple[list, list], sample_features: SpectralFeatures
    ) -> None:
        """Test that prediction returns ClassifierPrediction."""
        classifier = QualityClassifier()
        features_list, labels = training_features

        classifier.train(features_list, labels)
        prediction = classifier.predict(sample_features)

        assert prediction.format_type in ["128", "V2", "192", "V0", "256", "320", "LOSSLESS"]
        assert prediction.estimated_bitrate in [128, 190, 192, 245, 256, 320, 1411]
        assert 0 <= prediction.confidence <= 1
        assert isinstance(prediction.probabilities, dict)

    def test_save_and_load_model(
        self,
        training_features: tuple[list, list],
        temp_model_path,
        sample_features: SpectralFeatures,
    ) -> None:
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

    def test_predict_batch(
        self,
        training_features: tuple[list, list],
        sample_features: SpectralFeatures,
        lossless_features: SpectralFeatures,
    ) -> None:
        """Test batch prediction."""
        classifier = QualityClassifier()
        features_list, labels = training_features
        classifier.train(features_list, labels)

        predictions = classifier.predict_batch([sample_features, lossless_features])

        assert len(predictions) == 2
        valid_formats = {fmt for fmt, _ in BITRATE_CLASSES.values()}
        assert all(p.format_type in valid_formats for p in predictions)


class TestFeatureMask:
    """Tests for feature masking in QualityClassifier."""

    def test_feature_mask_applied_in_train(self, training_features: tuple[list, list]) -> None:
        """Training should apply feature mask, reducing feature count to 129."""
        classifier = QualityClassifier()
        features_list, labels = training_features

        classifier.train(features_list, labels)

        # Scaler should have seen 135 features (after mask)
        assert classifier.scaler.n_features_in_ == 135
        assert classifier.feature_mask is not None
        assert len(classifier.feature_mask) == 135

    def test_feature_mask_applied_in_predict(
        self, training_features: tuple[list, list], sample_features: SpectralFeatures
    ) -> None:
        """Prediction should apply same mask as training."""
        classifier = QualityClassifier()
        features_list, labels = training_features

        classifier.train(features_list, labels)
        prediction = classifier.predict(sample_features)

        # Should still produce valid prediction
        assert prediction.format_type in ["128", "V2", "192", "V0", "256", "320", "LOSSLESS"]

    def test_feature_mask_saved_and_loaded(
        self, training_features: tuple[list, list], temp_model_path, sample_features: SpectralFeatures
    ) -> None:
        """Feature mask should survive save/load cycle."""
        classifier1 = QualityClassifier()
        features_list, labels = training_features
        classifier1.train(features_list, labels, save_path=temp_model_path)

        pred1 = classifier1.predict(sample_features)

        classifier2 = QualityClassifier()
        classifier2.load_model(temp_model_path)

        assert classifier2.feature_mask is not None
        np.testing.assert_array_equal(classifier1.feature_mask, classifier2.feature_mask)

        pred2 = classifier2.predict(sample_features)
        assert pred1.format_type == pred2.format_type

    def test_feature_mask_count(self) -> None:
        """Feature mask should select 135 features from 173."""
        from bitrater.constants import FEATURE_MASK_NAMES, FEATURE_NAMES

        assert len(FEATURE_NAMES) == 173
        assert len(FEATURE_MASK_NAMES) == 135


class TestGridSearch:
    """Tests for grid_search method."""

    def test_grid_search_returns_results(self, training_features: tuple[list, list]) -> None:
        """Grid search should return results with best_params and best_score."""
        classifier = QualityClassifier()
        features_list, labels = training_features

        # Use small param grid for speed
        param_grid = {"kernel": ["poly"], "C": [1], "gamma": [1], "degree": [2]}
        results = classifier.grid_search(features_list, labels, param_grid=param_grid, cv=2)

        assert "best_params" in results
        assert "best_score" in results
        assert "elapsed_seconds" in results
        assert results["best_score"] > 0
        assert classifier.trained is True

    def test_grid_search_empty_data_raises(self) -> None:
        """Grid search with empty data should raise ValueError."""
        classifier = QualityClassifier()

        with pytest.raises(ValueError):
            classifier.grid_search([], [])

    def test_grid_search_saves_model(
        self, training_features: tuple[list, list], temp_model_path
    ) -> None:
        """Grid search should save model when save_path provided."""
        classifier = QualityClassifier()
        features_list, labels = training_features

        param_grid = {"kernel": ["poly"], "C": [1], "gamma": [1], "degree": [2]}
        classifier.grid_search(
            features_list, labels, param_grid=param_grid, cv=2, save_path=temp_model_path
        )

        assert temp_model_path.exists()

    def test_grid_search_applies_feature_mask(self, training_features: tuple[list, list]) -> None:
        """Grid search should apply feature mask before searching."""
        classifier = QualityClassifier()
        features_list, labels = training_features

        param_grid = {"kernel": ["poly"], "C": [1], "gamma": [1], "degree": [2]}
        classifier.grid_search(features_list, labels, param_grid=param_grid, cv=2)

        # Feature mask should be set and scaler should see 135 features
        assert classifier.feature_mask is not None
        assert classifier.scaler.n_features_in_ == 135


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
