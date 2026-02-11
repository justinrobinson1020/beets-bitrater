"""Audio quality analyzer - orchestrates spectral analysis and classification."""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

# Suppress numba FNV hashing warning (harmless, triggers once per worker process)
warnings.filterwarnings("ignore", message="FNV hashing is not implemented", module="numba")

# Import constants (lightweight, no numpy/scipy)
from .constants import (  # noqa: E402
    BITRATE_MISMATCH_FACTOR,
    CLASS_LABELS,
    LOSSLESS_CONTAINERS,
    LOW_CONFIDENCE_THRESHOLD,
)

# TYPE_CHECKING imports for type hints only - not imported at runtime in workers
if TYPE_CHECKING:
    from .file_analyzer import FileAnalyzer
    from .spectrum import SpectrumAnalyzer
    from .types import AnalysisResult, SpectralFeatures

logger = logging.getLogger("beets.bitrater")


# Module-level cache for worker instances (one per process)
_worker_initialized: bool = False
_worker_analyzer: SpectrumAnalyzer | None = None
_worker_file_analyzer: FileAnalyzer | None = None


def _init_worker() -> None:
    """One-time worker initialization: set thread limits before heavy imports."""
    global _worker_initialized
    if _worker_initialized:
        return

    from ._threading import clamp_threads_hard

    clamp_threads_hard()

    import numba

    numba.set_num_threads(1)

    _worker_initialized = True


@contextmanager
def _tqdm_joblib(tqdm_bar):
    """Context manager to patch joblib to update a tqdm progress bar on task completion."""
    original_print_progress = Parallel.print_progress

    def _patched_print_progress(self):
        tqdm_bar.n = self.n_completed_tasks
        tqdm_bar.refresh()

    Parallel.print_progress = _patched_print_progress
    try:
        yield
    finally:
        Parallel.print_progress = original_print_progress


def _extract_features_worker(file_path: str) -> tuple[str, SpectralFeatures | None]:
    """
    Worker function for joblib.Parallel - extracts features from audio file.

    CRITICAL: Reuses SpectrumAnalyzer/FileAnalyzer per process to avoid
    spawning new threads on each call. FeatureCache creates a worker thread
    that would leak if we created new instances per call.
    """
    global _worker_analyzer, _worker_file_analyzer

    # One-time setup per worker process
    _init_worker()

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        from .file_analyzer import FileAnalyzer
        from .spectrum import SpectrumAnalyzer

        # Create analyzers ONCE per worker process (avoids thread leak)
        if _worker_analyzer is None:
            _worker_analyzer = SpectrumAnalyzer()
            _worker_file_analyzer = FileAnalyzer()

        try:
            # Get metadata to determine is_vbr flag
            metadata = _worker_file_analyzer.analyze(file_path)
            is_vbr = 1.0 if metadata and metadata.encoding_type == "VBR" else 0.0

            features = _worker_analyzer.analyze_file(file_path, is_vbr=is_vbr)
            return (file_path, features)
        except FileNotFoundError:
            return (file_path, None)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Failed to extract features from {file_path}: {e}")
            return (file_path, None)
        except Exception as e:
            logger.error(
                f"Unexpected error extracting features from {file_path}: {e}", exc_info=True
            )
            return (file_path, None)


class AudioQualityAnalyzer:
    """
    Orchestrates audio quality analysis pipeline.

    Combines spectral analysis, SVM classification, and metadata examination
    to detect audio quality, verify lossless files, and identify transcodes.
    """

    def __init__(self, model_path: Path | None = None):
        """
        Initialize analyzer components.

        Args:
            model_path: Optional path to pre-trained classifier model
        """
        # Lazy imports - heavy libraries are imported here in main process,
        # AFTER env vars are set at module level. Workers use _extract_features_worker
        # which does its own lazy imports.
        from .classifier import QualityClassifier
        from .confidence import ConfidenceCalculator
        from .cutoff_detector import CutoffDetector
        from .file_analyzer import FileAnalyzer
        from .spectrum import SpectrumAnalyzer
        from .transcode_detector import TranscodeDetector

        self.spectrum_analyzer = SpectrumAnalyzer()
        self.classifier = QualityClassifier(model_path)
        self.file_analyzer = FileAnalyzer()
        self.cutoff_detector = CutoffDetector()
        self.confidence_calculator = ConfidenceCalculator()
        self.transcode_detector = TranscodeDetector()

    def analyze_file(self, file_path: str) -> AnalysisResult | None:
        """
        Analyze a single audio file with hybrid transcode detection.

        Performs spectral analysis, SVM classification, cutoff detection,
        and applies confidence penalties.
        """
        from .types import AnalysisResult

        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # 1. Get file metadata
        try:
            metadata = self.file_analyzer.analyze(file_path)
        except (ValueError, RuntimeError, KeyError) as e:
            # Expected errors: unsupported format, corrupt metadata, missing tags
            logger.warning(f"Could not read metadata from {file_path}: {e}")
            metadata = None
        except Exception as e:
            # Unexpected errors - log for investigation
            logger.error(
                f"Unexpected error reading metadata from {file_path}: {e}",
                exc_info=True,
            )
            metadata = None

        # 2. Extract spectral features with VBR metadata
        # Determine if file is VBR from metadata (helps discriminate V0/V2 from CBR)
        is_vbr = 0.0
        if metadata and metadata.encoding_type == "VBR":
            is_vbr = 1.0

        features = self.spectrum_analyzer.analyze_file(file_path, is_vbr=is_vbr)
        if features is None:
            logger.error(f"Failed to extract spectral features from {file_path}")
            return None

        # 3. Get file format and stated class
        file_format = path.suffix.lower().lstrip(".")
        stated_bitrate = metadata.bitrate if metadata else None
        stated_class = self._get_stated_class(file_format, stated_bitrate)

        # 4. Classify (if model is trained)
        if not self.classifier.trained:
            logger.warning("Classifier not trained - returning features without classification")
            return AnalysisResult(
                filename=str(path),
                file_format=file_format,
                original_format="UNKNOWN",
                original_bitrate=0,
                confidence=0.0,
                is_transcode=False,
                stated_class=stated_class,
                detected_cutoff=0,
                quality_gap=0,
                stated_bitrate=stated_bitrate,
                warnings=["Classifier not trained"],
            )

        prediction = self.classifier.predict(features)

        # 5. Cutoff detection for validation
        psd_data = self.spectrum_analyzer.get_psd(file_path)
        if psd_data is not None:
            psd, freqs = psd_data
            cutoff_result = self.cutoff_detector.detect(psd, freqs)
            detected_cutoff = cutoff_result.cutoff_frequency
            gradient = cutoff_result.gradient
        else:
            detected_cutoff = 0
            gradient = 0.5  # Neutral

        # 6. Calculate confidence with penalties
        conf_result = self.confidence_calculator.calculate(
            classifier_confidence=prediction.confidence,
            detected_class=prediction.format_type,
            detected_cutoff=detected_cutoff,
            gradient=gradient,
        )

        # 7. Detect transcode
        transcode_result = self.transcode_detector.detect(
            stated_class=stated_class,
            detected_class=prediction.format_type,
        )

        # 8. Collect all warnings
        warnings = list(conf_result.warnings)

        # Low confidence warning
        if conf_result.final_confidence < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(f"Low confidence in detection: {conf_result.final_confidence:.1%}")

        # Transcode warning
        if transcode_result.is_transcode:
            warnings.append(
                f"File appears to be transcoded from {transcode_result.transcoded_from} "
                f"(quality gap: {transcode_result.quality_gap})"
            )

        # Bitrate mismatch warning (stated vs detected for lossy files)
        if stated_bitrate and prediction.format_type != "LOSSLESS":
            detected_bitrate = prediction.estimated_bitrate
            if stated_bitrate > detected_bitrate * BITRATE_MISMATCH_FACTOR:
                warnings.append(
                    f"Stated bitrate ({stated_bitrate} kbps) much higher than "
                    f"detected ({detected_bitrate} kbps) - possible upsampled file"
                )

        return AnalysisResult(
            filename=str(path),
            file_format=file_format,
            original_format=prediction.format_type,
            original_bitrate=prediction.estimated_bitrate,
            confidence=conf_result.final_confidence,
            is_transcode=transcode_result.is_transcode,
            stated_class=stated_class,
            detected_cutoff=detected_cutoff,
            quality_gap=transcode_result.quality_gap,
            transcoded_from=transcode_result.transcoded_from,
            stated_bitrate=stated_bitrate,
            warnings=warnings,
        )

    def train(
        self,
        training_data: dict[str, int],
        save_path: Path | None = None,
    ) -> dict[str, int]:
        """
        Train the classifier from a dictionary of file paths and class labels.

        Args:
            training_data: Dictionary mapping file paths to class labels (0-5)
            save_path: Optional path to save the trained model

        Returns:
            Dictionary with training statistics
        """
        import time

        if not training_data:
            raise ValueError("No training data provided")

        features_list: list[SpectralFeatures] = []
        labels: list[int] = []
        failed_files: list[str] = []

        logger.info("=" * 60)
        logger.info(f"FEATURE EXTRACTION: Processing {len(training_data)} training files...")
        logger.info("=" * 60)

        extraction_start = time.time()
        # Log every file for maximum visibility during training
        progress_interval = 1

        for idx, (file_path, label) in enumerate(training_data.items(), 1):
            # Get metadata to determine is_vbr
            is_vbr = 0.0
            try:
                metadata = self.file_analyzer.analyze(file_path)
                if metadata and metadata.encoding_type == "VBR":
                    is_vbr = 1.0
            except (ValueError, RuntimeError, KeyError):
                # Expected errors: unsupported format, corrupt metadata, missing tags
                # Default to is_vbr=0.0 if metadata extraction fails
                pass
            except Exception as e:
                # Unexpected errors - log for investigation but continue
                logger.debug(f"Unexpected error reading metadata for {file_path}: {e}")

            features = self.spectrum_analyzer.analyze_file(file_path, is_vbr=is_vbr)
            if features is not None:
                features_list.append(features)
                labels.append(label)
            else:
                failed_files.append(file_path)
                logger.warning(f"Failed to extract features from: {file_path}")

            # Progress logging
            if idx % progress_interval == 0 or idx == len(training_data):
                pct = (idx / len(training_data)) * 100
                elapsed = time.time() - extraction_start
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (len(training_data) - idx) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {idx}/{len(training_data)} ({pct:.0f}%) - "
                    f"{rate:.1f} files/s - ETA: {eta:.0f}s"
                )

        extraction_time = time.time() - extraction_start

        if not features_list:
            raise ValueError("No valid training samples extracted")

        # Summary of feature extraction
        success_rate = len(features_list) / len(training_data) * 100
        logger.info("=" * 60)
        logger.info("FEATURE EXTRACTION COMPLETE")
        logger.info(f"  Total time: {extraction_time:.2f}s")
        logger.info(
            f"  Successful: {len(features_list)}/{len(training_data)} ({success_rate:.1f}%)"
        )
        logger.info(f"  Failed: {len(failed_files)}")
        logger.info(f"  Average: {extraction_time/len(training_data):.3f}s per file")
        if failed_files:
            logger.warning(f"  Failed files: {failed_files[:5]}")  # Show first 5
            if len(failed_files) > 5:
                logger.warning(f"  ... and {len(failed_files) - 5} more")
        logger.info("=" * 60)

        # Train classifier
        self.classifier.train(features_list, labels, save_path)

        return {
            "total_files": len(training_data),
            "successful": len(features_list),
            "failed": len(failed_files),
            "extraction_time": extraction_time,
        }

    def train_from_directory(
        self,
        training_dir: Path,
        save_path: Path | None = None,
        num_workers: int | None = None,
        use_parallel: bool | None = None,
    ) -> dict[str, int]:
        """
        Train the classifier from a directory structure.

        Supports two structures:

        Structure 1 (nested - preferred):
            training_dir/
            ├── encoded/
            │   └── lossy/
            │       ├── 128/
            │       ├── v2/
            │       ├── 192/
            │       ├── v0/
            │       ├── 256/
            │       └── 320/
            └── lossless/

        Structure 2 (flat):
            training_dir/
            ├── 128/
            ├── v2/
            ├── 192/
            ├── v0/
            ├── 256/
            ├── 320/
            └── lossless/

        Args:
            training_dir: Path to training data directory
            save_path: Optional path to save the trained model
            num_workers: Optional number of workers for parallel feature extraction
            use_parallel: Force parallel (True) or sequential (False). Defaults to True unless
                          num_workers == 1.

        Returns:
            Dictionary with training statistics
        """
        training_data = self._collect_training_data(Path(training_dir))

        # Decide on parallel vs sequential
        if use_parallel is None:
            use_parallel = num_workers != 1

        if use_parallel:
            return self.train_parallel(training_data, num_workers=num_workers, save_path=save_path)

        return self.train(training_data, save_path)

    def validate_from_directory(
        self,
        training_dir: Path,
        test_size: float = 0.2,
        random_state: int = 42,
        num_workers: int | None = None,
        use_parallel: bool | None = None,
    ) -> dict:
        """
        Validate the classifier using train/test split on a directory structure.

        Uses the same directory structure as train_from_directory. Splits data
        into training and test sets, trains on training data, and evaluates
        on test data.

        Args:
            training_dir: Path to training data directory
            test_size: Fraction of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            num_workers: Optional number of workers for parallel feature extraction
            use_parallel: Force parallel (True) or sequential (False). Defaults to True unless
                          num_workers == 1.

        Returns:
            Dictionary with validation metrics including:
            - accuracy: Overall accuracy
            - per_class: Per-class precision, recall, F1 scores
            - confusion_matrix: Confusion matrix
            - class_names: List of class names
            - total_samples: Total number of samples
            - train_samples: Number of training samples
            - test_samples: Number of test samples
        """
        import time

        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import train_test_split

        training_data = self._collect_training_data(Path(training_dir))

        # Split into train/test sets
        paths = list(training_data.keys())
        labels = [training_data[p] for p in paths]

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            paths, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        logger.info("=" * 60)
        logger.info("TRAIN/TEST SPLIT")
        logger.info(
            f"  Training set: {len(train_paths)} files ({len(train_paths)/len(paths)*100:.1f}%)"
        )
        logger.info(f"  Test set: {len(test_paths)} files ({len(test_paths)/len(paths)*100:.1f}%)")
        logger.info("=" * 60)

        # Extract features for ALL files once (train + test share the cache)
        if num_workers is None:
            workers = self._get_default_workers()
        else:
            workers = num_workers

        if use_parallel is None:
            use_parallel = workers != 1

        all_paths = train_paths + test_paths

        logger.info("=" * 60)
        logger.info("PARALLEL FEATURE EXTRACTION (joblib.Parallel)")
        logger.info(f"  Total files: {len(all_paths)}")
        logger.info(f"  Workers: {workers}")
        logger.info("=" * 60)

        extraction_start = time.time()

        if use_parallel:
            with tqdm(total=len(all_paths), desc="Extracting features", unit="files") as pbar:
                with _tqdm_joblib(pbar):
                    with parallel_config(backend="loky", inner_max_num_threads=1):
                        results = Parallel(n_jobs=workers, batch_size="auto", timeout=300)(
                            delayed(_extract_features_worker)(path)
                            for path in all_paths
                        )
        else:
            results = []
            for path in all_paths:
                results.append(_extract_features_worker(path))

        # Index results by path for fast lookup
        features_by_path: dict[str, SpectralFeatures | None] = {}
        for file_path, features in results:
            features_by_path[file_path] = features

        extraction_time = time.time() - extraction_start
        success_count = sum(1 for f in features_by_path.values() if f is not None)

        logger.info("=" * 60)
        logger.info("FEATURE EXTRACTION COMPLETE")
        logger.info(f"  Duration: {extraction_time:.2f}s")
        logger.info(
            f"  Success rate: {success_count/len(all_paths)*100:.1f}% ({success_count}/{len(all_paths)})"
        )
        logger.info(f"  Throughput: {success_count/extraction_time:.1f} files/s")
        logger.info("=" * 60)

        # Split extracted features into train/test
        train_features: list[SpectralFeatures] = []
        train_labels_clean: list[int] = []
        for path, label in zip(train_paths, train_labels, strict=True):
            feat = features_by_path.get(path)
            if feat is not None:
                train_features.append(feat)
                train_labels_clean.append(label)

        # Train classifier
        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info("=" * 60)
        self.classifier.train(train_features, train_labels_clean)

        # Predict on test set
        y_true = []
        y_pred = []
        failed_extractions = []

        logger.info("=" * 60)
        logger.info(f"VALIDATION: Running predictions on test set...")
        logger.info("=" * 60)

        predict_start = time.time()
        test_count = 0

        with tqdm(
            total=len(test_paths),
            desc="Predicting",
            unit="files",
            dynamic_ncols=True,
        ) as pbar:
            for path, true_label in zip(test_paths, test_labels, strict=True):
                feat = features_by_path.get(path)
                if feat is None:
                    failed_extractions.append(path)
                    pbar.update(1)
                    continue

                prediction = self.classifier.predict(feat)
                pred_label = CLASS_LABELS[prediction.format_type]
                y_true.append(true_label)
                y_pred.append(pred_label)
                test_count += 1

                elapsed = time.time() - predict_start
                rate = test_count / elapsed if elapsed > 0 else 0
                pbar.update(1)
                pbar.set_postfix({"rate": f"{rate:.1f} files/s"})

        predict_time = time.time() - predict_start
        total_time = extraction_time + predict_time

        logger.info("=" * 60)
        logger.info("VALIDATION COMPLETE")
        logger.info(f"  Feature extraction: {extraction_time:.2f}s")
        logger.info(f"  Predictions: {predict_time:.2f}s")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Successful: {len(y_true)}/{len(test_paths)}")
        logger.info(f"  Failed extractions: {len(failed_extractions)}")
        if failed_extractions:
            logger.warning(f"  Failed files: {failed_extractions[:3]}")
            if len(failed_extractions) > 3:
                logger.warning(f"  ... and {len(failed_extractions) - 3} more")
        logger.info("=" * 60)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(7)), zero_division=0
        )

        # Class names in order
        class_names = ["128", "V2", "192", "V0", "256", "320", "LOSSLESS"]

        # Build per-class metrics
        per_class = {}
        for i, cls in enumerate(class_names):
            per_class[cls] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }

        # Log detailed validation metrics
        logger.info("=" * 60)
        logger.info("VALIDATION METRICS")
        logger.info("=" * 60)
        logger.info(f"Overall Accuracy: {accuracy:.1%}")
        logger.info("")
        logger.info("Per-Class Performance:")
        logger.info(
            f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
        )
        logger.info("-" * 60)
        for cls in class_names:
            metrics = per_class[cls]
            if metrics["support"] > 0:
                logger.info(
                    f"{cls:<12} "
                    f"{metrics['precision']:>10.1%} "
                    f"{metrics['recall']:>10.1%} "
                    f"{metrics['f1']:>10.1%} "
                    f"{metrics['support']:>10}"
                )
        logger.info("")
        logger.info("Confusion Matrix:")
        logger.info(f"{'Actual/Pred':<12} " + " ".join(f"{name:>6}" for name in class_names))
        for i, actual_name in enumerate(class_names):
            if support[i] > 0:  # Only show rows with actual samples
                row_str = f"{actual_name:<12} " + " ".join(
                    f"{cm[i][j]:>6}" for j in range(len(class_names))
                )
                logger.info(row_str)
        logger.info("=" * 60)

        return {
            "accuracy": accuracy,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "total_samples": len(training_data),
            "train_samples": len(train_paths),
            "test_samples": len(test_paths),
            "train_pct": len(train_paths) / len(training_data),
            "test_pct": len(test_paths) / len(training_data),
        }


    def evaluate_from_directory(
        self,
        training_dir: Path,
        num_workers: int | None = None,
    ) -> dict:
        """
        Evaluate a pre-trained model on labeled data from a directory.

        Unlike validate_from_directory, this does NOT do a train/test split.
        It evaluates the already-loaded model on the full dataset.

        Args:
            training_dir: Path to labeled data directory
            num_workers: Optional number of workers for parallel feature extraction

        Returns:
            Dictionary with evaluation metrics (accuracy, per_class, etc.)
        """
        import time

        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
        )

        if not self.classifier.trained:
            raise RuntimeError("No model loaded. Use load_model() first.")

        training_data = self._collect_training_data(Path(training_dir))

        if num_workers is None:
            workers = self._get_default_workers()
        else:
            workers = num_workers

        file_paths = list(training_data.keys())

        logger.info("=" * 60)
        logger.info("MODEL EVALUATION: PARALLEL FEATURE EXTRACTION")
        logger.info(f"  Total files: {len(file_paths)}")
        logger.info(f"  Workers: {workers}")
        logger.info("=" * 60)

        extraction_start = time.time()

        with tqdm(total=len(file_paths), desc="Extracting features", unit="files") as pbar:
            with _tqdm_joblib(pbar):
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    results = Parallel(n_jobs=workers, batch_size="auto", timeout=300)(
                        delayed(_extract_features_worker)(path)
                        for path in file_paths
                    )

        extraction_time = time.time() - extraction_start

        # Run predictions
        y_true = []
        y_pred = []

        logger.info(f"Feature extraction: {extraction_time:.1f}s")
        logger.info(f"Running predictions on {len(results)} samples...")

        predict_start = time.time()
        for (file_path, features), true_label in zip(
            results, [training_data[p] for p in file_paths], strict=True
        ):
            if features is None:
                continue
            prediction = self.classifier.predict(features)
            pred_label = CLASS_LABELS[prediction.format_type]
            y_true.append(true_label)
            y_pred.append(pred_label)

        predict_time = time.time() - predict_start
        logger.info(f"Predictions: {predict_time:.1f}s, {len(y_true)} samples evaluated")

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(7)), zero_division=0
        )

        class_names = ["128", "V2", "192", "V0", "256", "320", "LOSSLESS"]
        per_class = {}
        for i, cls in enumerate(class_names):
            per_class[cls] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }

        return {
            "accuracy": accuracy,
            "per_class": per_class,
            "class_names": class_names,
            "total_samples": len(y_true),
        }

    def grid_search_from_directory(
        self,
        training_dir: Path,
        param_grid: dict | None = None,
        cv: int = 5,
        n_jobs: int = -1,
        num_workers: int | None = None,
        save_path: Path | None = None,
        progress_path: Path | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Run grid search over SVM hyperparameters using training data from a directory.

        Uses parallel feature extraction, then passes the full dataset to
        classifier.grid_search() (CV handles holdout internally).

        Args:
            training_dir: Path to training data directory
            param_grid: Parameter grid for GridSearchCV
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs for grid search
            num_workers: Workers for feature extraction
            save_path: Optional path to save the best model
            progress_path: Optional path for incremental progress save/resume

        Returns:
            Dict with best_params, best_score, elapsed_seconds, all_results
        """
        import time

        training_data = self._collect_training_data(Path(training_dir))

        # Determine workers
        if num_workers is None:
            workers = self._get_default_workers()
        else:
            workers = num_workers

        logger.info("=" * 60)
        logger.info("GRID SEARCH: PARALLEL FEATURE EXTRACTION")
        logger.info(f"  Total files: {len(training_data)}")
        logger.info(f"  Workers: {workers}")
        logger.info("=" * 60)

        extraction_start = time.time()

        file_paths = list(training_data.keys())
        with tqdm(total=len(file_paths), desc="Extracting features", unit="files") as pbar:
            with _tqdm_joblib(pbar):
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    results = Parallel(n_jobs=workers, batch_size="auto", timeout=300)(
                        delayed(_extract_features_worker)(file_path)
                        for file_path in file_paths
                    )

        features_list: list[SpectralFeatures] = []
        labels_list: list[int] = []
        failed_files: list[str] = []

        for file_path, features in results:
            label = training_data[file_path]
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
            else:
                failed_files.append(file_path)

        extraction_time = time.time() - extraction_start

        if not features_list:
            raise ValueError("No valid training samples extracted")

        logger.info(f"Feature extraction: {extraction_time:.1f}s, {len(features_list)} samples")

        return self.classifier.grid_search(
            features_list,
            labels_list,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            save_path=save_path,
            progress_path=progress_path,
            verbose=verbose,
        )

    def _collect_training_data(self, training_dir: Path) -> dict[str, int]:
        """Scan directory structure and return file-path-to-label mapping.

        Supports three directory layouts:

        1. Nested: training_dir/encoded/lossy/{128,v2,...}/ + training_dir/lossless/
        2. Semi-nested: training_dir/lossy/{128,v2,...}/ + training_dir/lossless/
        3. Flat: training_dir/{128,v2,...,lossless}/

        Args:
            training_dir: Root directory containing training data

        Returns:
            Dictionary mapping file paths (str) to class label indices (int)

        Raises:
            FileNotFoundError: If training_dir does not exist
            ValueError: If no audio files are found
        """
        training_dir = Path(training_dir)
        if not training_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {training_dir}")

        lossy_dir_to_class = {
            "128": CLASS_LABELS["128"],
            "v2": CLASS_LABELS["V2"],
            "192": CLASS_LABELS["192"],
            "v0": CLASS_LABELS["V0"],
            "256": CLASS_LABELS["256"],
            "320": CLASS_LABELS["320"],
        }

        training_data: dict[str, int] = {}
        audio_extensions = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus"}

        # Determine which structure we have
        nested_lossy_dir = training_dir / "encoded" / "lossy"
        semi_nested_lossy_dir = training_dir / "lossy"
        if nested_lossy_dir.exists():
            lossy_base = nested_lossy_dir
            lossless_dir = training_dir / "lossless"
            logger.info("Using nested directory structure (encoded/lossy/...)")
        elif semi_nested_lossy_dir.exists():
            lossy_base = semi_nested_lossy_dir
            lossless_dir = training_dir / "lossless"
            logger.info("Using semi-nested directory structure (lossy/...)")
        else:
            lossy_base = training_dir
            lossless_dir = training_dir / "lossless"
            logger.info("Using flat directory structure")

        # Collect lossy files
        for dir_name, class_label in lossy_dir_to_class.items():
            class_dir = lossy_base / dir_name
            if class_dir.exists():
                count = 0
                for file_path in class_dir.iterdir():
                    if file_path.suffix.lower() in audio_extensions:
                        training_data[str(file_path)] = class_label
                        count += 1
                logger.info(f"Found {count} files in {dir_name}/")
            else:
                logger.warning(f"Training directory not found: {class_dir}")

        # Collect lossless files
        if lossless_dir.exists():
            count = 0
            for file_path in lossless_dir.iterdir():
                if file_path.suffix.lower() in audio_extensions:
                    training_data[str(file_path)] = CLASS_LABELS["LOSSLESS"]
                    count += 1
            logger.info(f"Found {count} files in lossless/")
        else:
            logger.warning(f"Lossless directory not found: {lossless_dir}")

        if not training_data:
            raise ValueError(f"No training files found in {training_dir}")

        return training_data

    def _get_stated_class(self, file_format: str, stated_bitrate: int | None) -> str:
        """
        Determine stated quality class from file format and metadata.

        Args:
            file_format: Container format (mp3, flac, etc.)
            stated_bitrate: Bitrate from file metadata (if available)

        Returns:
            Quality class string: "128", "V2", "192", "V0", "256", "320", "LOSSLESS"
        """
        # Lossless containers are always stated as LOSSLESS
        if file_format in LOSSLESS_CONTAINERS:
            return "LOSSLESS"

        # For lossy containers, use bitrate to determine stated class
        if stated_bitrate is None:
            return "UNKNOWN"

        # Map bitrate to class
        if stated_bitrate <= 140:
            return "128"
        elif stated_bitrate <= 175:
            return "V2"
        elif stated_bitrate <= 210:
            return "192"
        elif stated_bitrate <= 260:
            return "V0"
        elif stated_bitrate <= 290:
            return "256"
        else:
            return "320"

    def save_model(self, path: Path) -> None:
        """Save the trained classifier model."""
        self.classifier.save_model(path)

    def load_model(self, path: Path) -> None:
        """Load a pre-trained classifier model."""
        self.classifier.load_model(path)

    @property
    def is_trained(self) -> bool:
        """Check if the classifier is trained."""
        return self.classifier.trained

    def _get_default_workers(self) -> int:
        """Get default number of workers based on CPU count (50% of available cores).

        Conservative default to prevent system overload. Heavy parallel I/O
        combined with CPU-intensive FFT work can stress the system.
        """
        cpu_count = os.cpu_count() or 1
        return max(1, int(cpu_count * 0.5))

    def train_parallel(
        self,
        training_data: dict[str, int],
        num_workers: int | None = None,
        save_path: Path | None = None,
    ) -> dict[str, int]:
        """
        Train the classifier using parallel feature extraction.

        Uses joblib.Parallel with loky backend for true parallelism. The loky
        backend automatically limits threads in worker processes to prevent
        oversubscription.

        Args:
            training_data: Dictionary mapping file paths to class labels (0-6)
            num_workers: Number of parallel workers (None = use CPU count)
            save_path: Optional path to save the trained model

        Returns:
            Dictionary with training statistics
        """
        import time

        if not training_data:
            raise ValueError("No training data provided")

        # Use conservative worker count to prevent system overload
        if num_workers is None:
            workers = self._get_default_workers()
        else:
            workers = num_workers

        logger.info("=" * 60)
        logger.info("PARALLEL FEATURE EXTRACTION (joblib.Parallel)")
        logger.info(f"  Total files: {len(training_data)}")
        logger.info(f"  Workers: {workers}")
        logger.info("=" * 60)

        extraction_start = time.time()

        # Process with joblib.Parallel
        # inner_max_num_threads=1 prevents thread explosion in workers
        file_paths = list(training_data.keys())
        with tqdm(total=len(file_paths), desc="Extracting features", unit="files") as pbar:
            with _tqdm_joblib(pbar):
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    results = Parallel(n_jobs=workers, batch_size="auto", timeout=300)(
                        delayed(_extract_features_worker)(file_path)
                        for file_path in file_paths
                    )

        # Process results
        features_list: list[SpectralFeatures] = []
        labels: list[int] = []
        failed_files: list[str] = []

        for file_path, features in results:
            label = training_data[file_path]
            if features is not None:
                features_list.append(features)
                labels.append(label)
            else:
                failed_files.append(file_path)

        extraction_time = time.time() - extraction_start

        if not features_list:
            raise ValueError("No valid training samples extracted")

        # Summary of extraction
        success_rate = len(features_list) / len(training_data) * 100
        logger.info("=" * 60)
        logger.info("FEATURE EXTRACTION COMPLETE")
        logger.info(f"  Duration: {extraction_time:.2f}s")
        logger.info(
            f"  Success rate: {success_rate:.1f}% ({len(features_list)}/{len(training_data)})"
        )
        logger.info(f"  Throughput: {len(features_list)/extraction_time:.1f} files/s")
        if failed_files:
            logger.warning(f"  Failed: {len(failed_files)} files")
        logger.info("=" * 60)

        # Train classifier (this will log its own metrics)
        self.classifier.train(features_list, labels, save_path)

        return {
            "total_files": len(training_data),
            "successful": len(features_list),
            "failed": len(failed_files),
            "extraction_time": extraction_time,
            "workers": workers,
            "throughput": len(features_list) / extraction_time if extraction_time > 0 else 0,
        }
