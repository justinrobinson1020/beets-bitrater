"""Audio quality classification using SVM.

Based on D'Alessandro & Shi paper methodology with polynomial SVM.
"""

import logging
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .constants import BITRATE_CLASSES, CLASSIFIER_PARAMS, FEATURE_MASK_NAMES, FEATURE_NAMES
from .types import ClassifierPrediction, SpectralFeatures

logger = logging.getLogger("beets.bitrater")


def _save_grid_progress(progress_path: Path, all_results: list, best_params: dict, best_score: float) -> None:
    """Atomically save grid search progress to JSON."""
    import json
    import tempfile

    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "best_params": best_params,
        "best_score": best_score,
        "completed": len(all_results),
        "all_results": all_results,
    }
    # Atomic write: write to temp file then rename
    fd, tmp = tempfile.mkstemp(dir=progress_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, progress_path)
    except BaseException:
        os.unlink(tmp)
        raise


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
        self.feature_mask: np.ndarray | None = None

        # Use predefined classes from constants
        self.classes = BITRATE_CLASSES

        # Load pre-trained model if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    @staticmethod
    def _resolve_feature_mask(keep_names: set[str]) -> np.ndarray:
        """Resolve feature mask from a set of feature names to keep.

        Returns an array of indices into the 167-feature vector.
        """
        return np.array(
            [i for i, name in enumerate(FEATURE_NAMES) if name in keep_names],
            dtype=np.intp,
        )

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

        # Apply feature mask
        self.feature_mask = self._resolve_feature_mask(FEATURE_MASK_NAMES)
        X = X[:, self.feature_mask]
        logger.info(f"Feature mask applied: {len(FEATURE_NAMES)} -> {X.shape[1]} features")

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
        logger.info(
            f"  Support vectors per class: {dict(zip([self.classes[c][0] for c in unique], self.classifier.n_support_, strict=True))}"
        )

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

    def grid_search(
        self,
        features_list: list[SpectralFeatures],
        labels: list[int],
        param_grid: dict | None = None,
        cv: int = 5,
        n_jobs: int = -1,
        save_path: Path | None = None,
        progress_path: Path | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Run grid search over SVM hyperparameters.

        Args:
            features_list: List of SpectralFeatures objects
            labels: List of class labels
            param_grid: Parameter grid for GridSearchCV. If None, uses default.
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 = all CPUs)
            save_path: Optional path to save the best model
            progress_path: Optional path to save/resume incremental progress (JSON)
            verbose: If True, log per-combination timing and per-fold scores

        Returns:
            Dict with best_params, best_score, elapsed_seconds, all_results
        """
        import json
        import time

        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        if not features_list or not labels:
            raise ValueError("Empty training data")

        # Extract features
        X = np.array([self._extract_features(f) for f in features_list])
        y = np.array(labels)

        # Apply feature mask
        self.feature_mask = self._resolve_feature_mask(FEATURE_MASK_NAMES)
        X = X[:, self.feature_mask]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                "kernel": ["poly"],
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", 0.01, 0.1, 1, 10],
                "degree": [2, 3],
            }

        # Calculate total work for progress reporting
        from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
        from tqdm import tqdm

        param_list = list(ParameterGrid(param_grid))
        n_candidates = len(param_list)
        n_fits = n_candidates * cv

        logger.info("=" * 60)
        logger.info("GRID SEARCH")
        logger.info(f"  Samples: {len(X)}")
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  CV folds: {cv}")
        logger.info(f"  Parameter combinations: {n_candidates}")
        logger.info(f"  Total fits: {n_fits}")
        logger.info(f"  Parallel jobs: {n_jobs}")
        logger.info(f"  Param grid: {param_grid}")
        logger.info("=" * 60)

        # Set up CV strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Base SVM params shared across all candidates
        base_params = {
            "coef0": CLASSIFIER_PARAMS["coef0"],
            "probability": True,
            "cache_size": CLASSIFIER_PARAMS["cache_size"],
            "class_weight": CLASSIFIER_PARAMS["class_weight"],
            "random_state": CLASSIFIER_PARAMS["random_state"],
        }

        # Load existing progress if available
        completed: dict[str, dict] = {}
        if progress_path and Path(progress_path).exists():
            with open(progress_path) as f:
                saved = json.load(f)
            for r in saved.get("all_results", []):
                key = json.dumps(r["params"], sort_keys=True)
                completed[key] = r
            logger.info(f"Resuming grid search: {len(completed)}/{n_candidates} already done")

        # Manual grid search with tqdm progress bar
        all_results = list(completed.values())
        best_score = max((r["mean_score"] for r in all_results), default=-1.0)
        best_params = {}
        if all_results and best_score > 0:
            best_params = max(all_results, key=lambda r: r["mean_score"])["params"]

        start = time.time()
        skipped = 0
        with tqdm(
            total=n_candidates,
            initial=len(completed),
            desc="Grid search",
            unit="combo",
            dynamic_ncols=True,
        ) as pbar:
            for i, params in enumerate(param_list):
                key = json.dumps(params, sort_keys=True)
                if key in completed:
                    skipped += 1
                    continue

                if verbose:
                    logger.info(f"  [{i+1}/{n_candidates}] Testing {params} ...")

                svm = SVC(**base_params, **params, max_iter=50_000)
                combo_start = time.time()
                scores = cross_val_score(
                    svm, X_scaled, y, cv=cv_strategy, scoring="accuracy", n_jobs=n_jobs
                )
                combo_elapsed = time.time() - combo_start
                mean_score = float(scores.mean())
                std_score = float(scores.std())

                if verbose:
                    fold_str = " ".join(f"{s:.4f}" for s in scores)
                    logger.info(
                        f"  [{i+1}/{n_candidates}] {params} -> {mean_score:.4f} +/- {std_score:.4f} "
                        f"[{combo_elapsed:.1f}s] folds: [{fold_str}]"
                    )

                result = {
                    "params": params,
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "elapsed": combo_elapsed,
                }
                all_results.append(result)
                completed[key] = result

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

                pbar.update(1)
                pbar.set_postfix({
                    "best": f"{best_score:.4f}",
                    "current": f"{mean_score:.4f}",
                    "time": f"{combo_elapsed:.1f}s",
                })

                # Save progress after each combination
                if progress_path:
                    _save_grid_progress(progress_path, all_results, best_params, best_score)

        elapsed = time.time() - start

        # Refit best model on full dataset
        best_estimator = SVC(**base_params, **best_params, max_iter=50_000)
        best_estimator.fit(X_scaled, y)
        self.classifier = best_estimator
        self.trained = True

        # Rank results
        all_results.sort(key=lambda r: r["mean_score"], reverse=True)
        for rank, r in enumerate(all_results, 1):
            r["rank"] = rank

        logger.info("=" * 60)
        logger.info(f"GRID SEARCH COMPLETE in {elapsed:.1f}s")
        logger.info(f"  Best score: {best_score:.4f}")
        logger.info(f"  Best params: {best_params}")
        logger.info("=" * 60)

        if save_path:
            self.save_model(save_path)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "elapsed_seconds": elapsed,
            "all_results": all_results,
        }

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
        if self.feature_mask is not None:
            X = X[:, self.feature_mask]
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

    def predict_batch(self, features_list: list[SpectralFeatures]) -> list[ClassifierPrediction]:
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

        # Store mask as list of kept feature names (robust to future reordering)
        if self.feature_mask is not None:
            mask_names = [FEATURE_NAMES[i] for i in self.feature_mask]
        else:
            mask_names = None

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "classifier": self.classifier,
                    "scaler": self.scaler,
                    "classes": self.classes,
                    "trained": self.trained,
                    "feature_mask_names": mask_names,
                    "version": "5.0",
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

        # Restore feature mask from saved names
        mask_names = data.get("feature_mask_names")
        if mask_names is not None:
            self.feature_mask = self._resolve_feature_mask(set(mask_names))
        else:
            self.feature_mask = None

        # Version check
        model_version = data.get("version")
        if model_version != "5.0":
            logger.warning(
                f"Model version mismatch: expected '5.0', got '{model_version}'. "
                "Predictions may be unreliable."
            )

        # Log feature count from scaler
        n_features = getattr(self.scaler, "n_features_in_", None)
        if n_features is not None:
            logger.info(f"Model loaded from {path} ({n_features} features)")
        else:
            logger.info(f"Model loaded from {path}")
