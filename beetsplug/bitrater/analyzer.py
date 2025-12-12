"""Audio quality analyzer - orchestrates spectral analysis and classification."""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .classifier import QualityClassifier
from .confidence import ConfidenceCalculator
from .constants import (
    BITRATE_MISMATCH_FACTOR,
    CLASS_LABELS,
    LOSSLESS_CONTAINERS,
    LOW_CONFIDENCE_THRESHOLD,
)
from .cutoff_detector import CutoffDetector
from .file_analyzer import FileAnalyzer
from .spectrum import SpectrumAnalyzer
from .transcode_detector import TranscodeDetector
from .types import AnalysisResult, SpectralFeatures

logger = logging.getLogger("beets.bitrater")


def _extract_features_worker(file_path: str) -> tuple[str, SpectralFeatures | None]:
    """
    Worker function for ProcessPoolExecutor - extracts features from audio file.

    Must be at module level (not a method) to be picklable by multiprocessing.
    Creates its own SpectrumAnalyzer and FileAnalyzer instances to avoid
    sharing state across processes.

    Args:
        file_path: Path to audio file to analyze

    Returns:
        Tuple of (file_path, features) where features is None if extraction failed
    """
    try:
        analyzer = SpectrumAnalyzer()
        file_analyzer = FileAnalyzer()

        # Get metadata to determine is_vbr flag for accurate training labels
        metadata = file_analyzer.analyze(file_path)
        is_vbr = 1.0 if metadata and metadata.encoding_type == "VBR" else 0.0

        features = analyzer.analyze_file(file_path, is_vbr=is_vbr)
        return (file_path, features)
    except FileNotFoundError:
        # File doesn't exist - expected in some cases, don't log
        return (file_path, None)
    except (ValueError, RuntimeError) as e:
        # Audio format errors or analysis failures
        logger.warning(f"Failed to extract features from {file_path}: {e}")
        return (file_path, None)
    except Exception as e:
        # Unexpected errors - log for investigation
        logger.error(f"Unexpected error extracting features from {file_path}: {e}", exc_info=True)
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
            warnings.append(
                f"Low confidence in detection: {conf_result.final_confidence:.1%}"
            )

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
        logger.info(f"  Successful: {len(features_list)}/{len(training_data)} ({success_rate:.1f}%)")
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
        training_dir = Path(training_dir)
        if not training_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {training_dir}")

        # Map directory names to class labels for lossy classes
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
        if nested_lossy_dir.exists():
            # Structure 1: nested structure
            lossy_base = nested_lossy_dir
            lossless_dir = training_dir / "lossless"
            logger.info("Using nested directory structure (encoded/lossy/...)")
        else:
            # Structure 2: flat structure
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

        # Decide on parallel vs sequential
        if use_parallel is None:
            use_parallel = num_workers != 1

        if use_parallel:
            return self.train_parallel(training_data, num_workers=num_workers, save_path=save_path)

        return self.train(training_data, save_path)

    def validate_from_directory(
        self,
        training_dir: Path,
        test_size: float = 0.8,
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
            test_size: Fraction of data to use for testing (default: 0.8)
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

        training_dir = Path(training_dir)
        if not training_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {training_dir}")

        # Map directory names to class labels for lossy classes
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
        if nested_lossy_dir.exists():
            lossy_base = nested_lossy_dir
            lossless_dir = training_dir / "lossless"
            logger.info("Using nested directory structure (encoded/lossy/...)")
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

        # Collect lossless files
        if lossless_dir.exists():
            count = 0
            for file_path in lossless_dir.iterdir():
                if file_path.suffix.lower() in audio_extensions:
                    training_data[str(file_path)] = CLASS_LABELS["LOSSLESS"]
                    count += 1
            logger.info(f"Found {count} files in lossless/")

        if not training_data:
            raise ValueError(f"No training files found in {training_dir}")

        # Split into train/test sets
        paths = list(training_data.keys())
        labels = [training_data[p] for p in paths]

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            paths, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        logger.info("=" * 60)
        logger.info("TRAIN/TEST SPLIT")
        logger.info(f"  Training set: {len(train_paths)} files ({len(train_paths)/len(paths)*100:.1f}%)")
        logger.info(f"  Test set: {len(test_paths)} files ({len(test_paths)/len(paths)*100:.1f}%)")
        logger.info("=" * 60)

        # Train on training set
        train_data = dict(zip(train_paths, train_labels, strict=True))
        if use_parallel is None:
            use_parallel = num_workers != 1

        if use_parallel:
            self.train_parallel(train_data, num_workers=num_workers)
        else:
            self.train(train_data)

        # Evaluate on test set
        y_true = []
        y_pred = []
        failed_predictions = []

        logger.info("=" * 60)
        logger.info(f"VALIDATION: Evaluating on {len(test_paths)} test samples...")
        logger.info("=" * 60)

        eval_start = time.time()
        progress_interval = max(1, len(test_paths) // 10)  # Log every 10%

        for idx, (path, true_label) in enumerate(zip(test_paths, test_labels, strict=True), 1):
            features = self.spectrum_analyzer.analyze_file(path)
            if features:
                pred_label, _ = self.classifier.predict(features)
                y_true.append(true_label)
                y_pred.append(pred_label)
            else:
                failed_predictions.append(path)
                logger.warning(f"Failed to extract features for validation: {path}")

            # Progress logging
            if idx % progress_interval == 0 or idx == len(test_paths):
                pct = (idx / len(test_paths)) * 100
                elapsed = time.time() - eval_start
                rate = idx / elapsed if elapsed > 0 else 0
                logger.info(f"Validation progress: {idx}/{len(test_paths)} ({pct:.0f}%) - {rate:.1f} files/s")

        eval_time = time.time() - eval_start

        logger.info("=" * 60)
        logger.info("VALIDATION EVALUATION COMPLETE")
        logger.info(f"  Total time: {eval_time:.2f}s")
        logger.info(f"  Successful: {len(y_true)}/{len(test_paths)}")
        logger.info(f"  Failed: {len(failed_predictions)}")
        logger.info(f"  Average: {eval_time/len(test_paths):.3f}s per file")
        if failed_predictions:
            logger.warning(f"  Failed predictions: {failed_predictions[:3]}")
            if len(failed_predictions) > 3:
                logger.warning(f"  ... and {len(failed_predictions) - 3} more")
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
        logger.info(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        logger.info("-" * 60)
        for cls in class_names:
            metrics = per_class[cls]
            if metrics['support'] > 0:
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
                row_str = f"{actual_name:<12} " + " ".join(f"{cm[i][j]:>6}" for j in range(len(class_names)))
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
        """Get default number of workers based on CPU count (80% of available cores)."""
        cpu_count = os.cpu_count() or 1
        return max(1, int(cpu_count * 0.8))

    def train_parallel(
        self,
        training_data: dict[str, int],
        num_workers: int | None = None,
        save_path: Path | None = None,
    ) -> dict[str, int]:
        """
        Train the classifier using parallel feature extraction with batch processing.

        Uses ProcessPoolExecutor for true parallelism (bypasses GIL) and processes
        files in batches to avoid memory accumulation. Provides live progress bar
        using tqdm.

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

        # Use all available CPU cores for true parallelism
        if num_workers is None:
            workers = os.cpu_count() or 1
        else:
            workers = num_workers

        features_list: list[SpectralFeatures] = []
        labels: list[int] = []
        failed_files: list[str] = []

        logger.info("=" * 60)
        logger.info("PARALLEL FEATURE EXTRACTION (ProcessPoolExecutor)")
        logger.info(f"  Total files: {len(training_data)}")
        logger.info(f"  Workers: {workers}")
        logger.info("=" * 60)

        extraction_start = time.time()

        # Process with ProcessPoolExecutor for CPU-bound FFT work
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks using module-level worker function (picklable)
            futures = {
                executor.submit(_extract_features_worker, file_path): (file_path, label)
                for file_path, label in training_data.items()
            }

            # Use tqdm for live progress feedback
            with tqdm(
                total=len(training_data),
                desc="Extracting features",
                unit="files",
                dynamic_ncols=True,
            ) as pbar:
                for future in futures:
                    file_path, features = future.result()
                    label = futures[future][1]

                    if features is not None:
                        features_list.append(features)
                        labels.append(label)
                    else:
                        failed_files.append(file_path)

                    # Update progress bar
                    elapsed = time.time() - extraction_start
                    rate = len(features_list) / elapsed if elapsed > 0 else 0
                    pbar.update(1)
                    pbar.set_postfix({
                        "rate": f"{rate:.1f} files/s",
                        "failed": len(failed_files),
                    })

        extraction_time = time.time() - extraction_start

        if not features_list:
            raise ValueError("No valid training samples extracted")

        # Summary of extraction
        success_rate = len(features_list) / len(training_data) * 100
        logger.info("=" * 60)
        logger.info("FEATURE EXTRACTION COMPLETE")
        logger.info(f"  Duration: {extraction_time:.2f}s")
        logger.info(f"  Success rate: {success_rate:.1f}% ({len(features_list)}/{len(training_data)})")
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
