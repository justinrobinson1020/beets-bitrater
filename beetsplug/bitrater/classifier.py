"""Audio quality classification using SVM.

Based on D'Alessandro & Shi paper methodology with polynomial SVM.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .constants import BITRATE_CLASSES, CLASSIFIER_PARAMS
from .types import ClassifierPrediction, SpectralFeatures

logger = logging.getLogger("beets.bitrater")


class QualityClassifier:
    """
    Classifies audio quality using SVM on PSD frequency band features.

    Uses polynomial SVM (degree=2) as per D'Alessandro & Shi paper.
    Classifies into 6 classes: 128, 192, 256, 320, V0, LOSSLESS.
    """

    def __init__(self, model_path: Path | None = None):
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

        Returns concatenated vector: PSD bands + cutoff + temporal + artifact + is_vbr.
        The is_vbr feature helps discriminate VBR (V0, V2) from CBR (128, 192, etc).
        """
        return features.as_vector().astype(np.float32)

    def train(
        self,
        features_list: list[SpectralFeatures],
        labels: list[int],
        save_path: Path | None = None,
    ) -> None:
        """
        Train the classifier on spectral features.

        Args:
            features_list: List of SpectralFeatures objects
            labels: List of class labels (0-5 corresponding to BITRATE_CLASSES)
            save_path: Optional path to save trained model
        """
        import time

        train_start = time.time()

        if not features_list or not labels:
            raise ValueError("Empty training data")
        if len(features_list) != len(labels):
            raise ValueError(
                f"Mismatched lengths: {len(features_list)} features, {len(labels)} labels"
            )

        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info("=" * 60)

        # Log model hyperparameters
        logger.info("Model hyperparameters:")
        logger.info(f"  Kernel: {self.classifier.kernel}")
        logger.info(f"  Degree: {self.classifier.degree}")
        logger.info(f"  Gamma: {self.classifier.gamma}")
        logger.info(f"  C: {self.classifier.C}")
        logger.info(f"  Class weight: {self.classifier.class_weight}")

        # Extract features with progress logging
        logger.info(f"Extracting features from {len(features_list)} samples...")
        extract_start = time.time()
        X = np.array([self._extract_features(f) for f in features_list])
        y = np.array(labels)
        extract_time = time.time() - extract_start
        logger.info(f"Feature extraction completed in {extract_time:.2f}s")

        # Log feature matrix statistics
        logger.info("Feature matrix statistics:")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Min: {X.min():.6f}")
        logger.info(f"  Max: {X.max():.6f}")
        logger.info(f"  Mean: {X.mean():.6f}")
        logger.info(f"  Std: {X.std():.6f}")
        logger.info(f"  NaN count: {np.isnan(X).sum()}")
        logger.info(f"  Inf count: {np.isinf(X).sum()}")

        # Validate labels
        valid_labels = set(self.classes.keys())
        invalid = set(y) - valid_labels
        if invalid:
            raise ValueError(f"Invalid labels: {invalid}. Must be in {valid_labels}")

        # Log class distribution and check for imbalance
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {self.classes[c][0]: n for c, n in zip(unique, counts, strict=True)}
        logger.info("Class distribution:")
        for class_name, count in class_dist.items():
            pct = count / len(y) * 100
            logger.info(f"  {class_name}: {count} samples ({pct:.1f}%)")

        # Check for class imbalance
        if len(counts) > 1:
            imbalance_ratio = counts.max() / counts.min()
            if imbalance_ratio > 3:
                logger.warning(
                    f"Class imbalance detected: ratio {imbalance_ratio:.1f}:1 "
                    f"(max={counts.max()}, min={counts.min()})"
                )
            else:
                logger.info(f"Class balance ratio: {imbalance_ratio:.1f}:1")

        # Check for missing classes
        missing_classes = valid_labels - set(y)
        if missing_classes:
            missing_names = [self.classes[c][0] for c in missing_classes]
            logger.warning(f"Missing classes in training data: {missing_names}")

        # Scale features
        logger.info("Scaling features...")
        scale_start = time.time()
        X_scaled = self.scaler.fit_transform(X)
        scale_time = time.time() - scale_start
        logger.info(f"Scaling completed in {scale_time:.3f}s")
        logger.info("Scaled feature statistics:")
        logger.info(f"  Mean: {X_scaled.mean():.6f} (should be ~0)")
        logger.info(f"  Std: {X_scaled.std():.6f} (should be ~1)")

        # Train classifier
        logger.info("Training SVM classifier...")
        fit_start = time.time()
        self.classifier.fit(X_scaled, y)
        fit_time = time.time() - fit_start
        self.trained = True
        logger.info(f"SVM training completed in {fit_time:.2f}s")

        # Log model info
        logger.info("Trained model info:")
        logger.info(f"  Support vectors: {self.classifier.n_support_.sum()}")
        logger.info(f"  Support vectors per class: {dict(zip([self.classes[c][0] for c in unique], self.classifier.n_support_, strict=True))}")

        # Total training time
        total_time = time.time() - train_start
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETED in {total_time:.2f}s")
        logger.info(f"  Samples: {len(X)}")
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  Classes: {len(unique)}")
        logger.info("=" * 60)

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
        self, features_list: list[SpectralFeatures]
    ) -> list[ClassifierPrediction]:
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
