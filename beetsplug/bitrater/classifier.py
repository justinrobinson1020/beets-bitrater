"""Audio quality classification using SVM.

Based on D'Alessandro & Shi paper methodology with polynomial SVM.
"""

from pathlib import Path
import pickle
import numpy as np
from typing import List, Optional, Dict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import logging

from .types import SpectralFeatures, ClassifierPrediction
from .constants import BITRATE_CLASSES, CLASSIFIER_PARAMS

logger = logging.getLogger(__name__)


class QualityClassifier:
    """
    Classifies audio quality using SVM on PSD frequency band features.

    Uses polynomial SVM (degree=2) as per D'Alessandro & Shi paper.
    Classifies into 6 classes: 128, 192, 256, 320, V0, LOSSLESS.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize classifier with paper's SVM parameters.

        Args:
            model_path: Optional path to saved model file
        """
        # Initialize SVM with ALL parameters from paper
        self.classifier = SVC(
            kernel=CLASSIFIER_PARAMS["kernel"],
            degree=CLASSIFIER_PARAMS["degree"],  # d=2 - CRITICAL
            gamma=CLASSIFIER_PARAMS["gamma"],  # γ=1
            C=CLASSIFIER_PARAMS["C"],  # C=1
            coef0=CLASSIFIER_PARAMS["coef0"],  # Standard for poly
            probability=CLASSIFIER_PARAMS["probability"],
            cache_size=CLASSIFIER_PARAMS["cache_size"],
            class_weight=CLASSIFIER_PARAMS["class_weight"],
            random_state=CLASSIFIER_PARAMS["random_state"],
        )

        self.scaler = StandardScaler()
        self.trained = False

        # Use predefined classes from constants
        self.classes = BITRATE_CLASSES

        # Load pre-trained model if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def _extract_features(self, features: SpectralFeatures) -> np.ndarray:
        """
        Extract feature vector from SpectralFeatures.

        Simply returns the 150 PSD band values - no additional processing.
        """
        return np.asarray(features.features, dtype=np.float32)

    def train(
        self,
        features_list: List[SpectralFeatures],
        labels: List[int],
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Train the classifier on spectral features.

        Args:
            features_list: List of SpectralFeatures objects
            labels: List of class labels (0-5 corresponding to BITRATE_CLASSES)
            save_path: Optional path to save trained model
        """
        if not features_list or not labels:
            raise ValueError("Empty training data")
        if len(features_list) != len(labels):
            raise ValueError(
                f"Mismatched lengths: {len(features_list)} features, {len(labels)} labels"
            )

        # Extract features
        X = np.array([self._extract_features(f) for f in features_list])
        y = np.array(labels)

        # Validate labels
        valid_labels = set(self.classes.keys())
        invalid = set(y) - valid_labels
        if invalid:
            raise ValueError(f"Invalid labels: {invalid}. Must be in {valid_labels}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        self.classifier.fit(X_scaled, y)
        self.trained = True

        # Log training info
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {self.classes[c][0]: n for c, n in zip(unique, counts)}
        logger.info(f"Trained on {len(X)} samples. Class distribution: {class_dist}")

        # Save model if requested
        if save_path:
            self.save_model(save_path)

    def predict(self, features: SpectralFeatures) -> ClassifierPrediction:
        """
        Predict quality class from spectral features.

        Args:
            features: SpectralFeatures object with 150 PSD band values

        Returns:
            ClassifierPrediction with format type, bitrate, and confidence
        """
        if not self.trained:
            raise RuntimeError("Classifier must be trained before prediction")

        # Extract and scale features
        X = self._extract_features(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Get prediction and probabilities
        predicted_class = self.classifier.predict(X_scaled)[0]
        class_probs = self.classifier.predict_proba(X_scaled)[0]

        # Get format name and bitrate from class mapping
        format_type, estimated_bitrate = self.classes[predicted_class]
        confidence = float(class_probs[predicted_class])

        # Create probability dictionary mapping bitrate to probability
        probabilities = {}
        for class_idx, prob in enumerate(class_probs):
            if class_idx in self.classes:
                _, bitrate = self.classes[class_idx]
                probabilities[bitrate] = float(prob)

        return ClassifierPrediction(
            format_type=format_type,
            estimated_bitrate=estimated_bitrate,
            confidence=confidence,
            probabilities=probabilities,
        )

    def predict_batch(
        self, features_list: List[SpectralFeatures]
    ) -> List[ClassifierPrediction]:
        """
        Predict quality class for multiple files.

        Args:
            features_list: List of SpectralFeatures objects

        Returns:
            List of ClassifierPrediction objects
        """
        return [self.predict(f) for f in features_list]

    def save_model(self, path: Path) -> None:
        """Save trained model and scaler to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "classifier": self.classifier,
                    "scaler": self.scaler,
                    "classes": self.classes,
                    "trained": self.trained,
                    "version": "3.0",
                },
                f,
            )
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """Load trained model and scaler from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.classifier = data["classifier"]
            self.scaler = data["scaler"]
            self.classes = data.get("classes", BITRATE_CLASSES)
            self.trained = data["trained"]

        logger.info(f"Model loaded from {path}")
