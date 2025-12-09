"""Audio quality analyzer - orchestrates spectral analysis and classification."""

from pathlib import Path
from typing import Optional, Dict, List
import logging

from .spectrum import SpectrumAnalyzer
from .classifier import QualityClassifier
from .file_analyzer import FileAnalyzer
from .cutoff_detector import CutoffDetector
from .confidence import ConfidenceCalculator
from .transcode_detector import TranscodeDetector
from .types import AnalysisResult, SpectralFeatures
from .constants import LOSSLESS_CONTAINERS, CLASS_LABELS

logger = logging.getLogger(__name__)


class AudioQualityAnalyzer:
    """
    Orchestrates audio quality analysis pipeline.

    Combines spectral analysis, SVM classification, and metadata examination
    to detect audio quality, verify lossless files, and identify transcodes.
    """

    def __init__(self, model_path: Optional[Path] = None):
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

    def analyze_file(self, file_path: str) -> Optional[AnalysisResult]:
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
        except Exception as e:
            logger.warning(f"Could not read metadata from {file_path}: {e}")
            metadata = None

        # 2. Extract spectral features
        features = self.spectrum_analyzer.analyze_file(file_path)
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
        if transcode_result.is_transcode:
            warnings.append(
                f"File appears to be transcoded from {transcode_result.transcoded_from} "
                f"(quality gap: {transcode_result.quality_gap})"
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
        training_data: Dict[str, int],
        save_path: Optional[Path] = None,
    ) -> Dict[str, int]:
        """
        Train the classifier from a dictionary of file paths and class labels.

        Args:
            training_data: Dictionary mapping file paths to class labels (0-5)
            save_path: Optional path to save the trained model

        Returns:
            Dictionary with training statistics
        """
        if not training_data:
            raise ValueError("No training data provided")

        features_list: List[SpectralFeatures] = []
        labels: List[int] = []
        failed_files: List[str] = []

        logger.info(f"Processing {len(training_data)} training files...")

        for file_path, label in training_data.items():
            features = self.spectrum_analyzer.analyze_file(file_path)
            if features is not None:
                features_list.append(features)
                labels.append(label)
            else:
                failed_files.append(file_path)
                logger.warning(f"Failed to extract features from: {file_path}")

        if not features_list:
            raise ValueError("No valid training samples extracted")

        logger.info(
            f"Extracted features from {len(features_list)} files "
            f"({len(failed_files)} failed)"
        )

        # Train classifier
        self.classifier.train(features_list, labels, save_path)

        return {
            "total_files": len(training_data),
            "successful": len(features_list),
            "failed": len(failed_files),
        }

    def train_from_directory(
        self,
        training_dir: Path,
        save_path: Optional[Path] = None,
    ) -> Dict[str, int]:
        """
        Train the classifier from a directory structure.

        Expected structure:
            training_dir/
            ├── 128/      # Files encoded at 128 kbps (class 0)
            ├── v2/       # VBR-2 files (class 1)
            ├── 192/      # Files encoded at 192 kbps (class 2)
            ├── v0/       # VBR-0 files (class 3)
            ├── 256/      # Files encoded at 256 kbps (class 4)
            ├── 320/      # Files encoded at 320 kbps (class 5)
            └── lossless/ # Lossless files (class 6)

        Args:
            training_dir: Path to training data directory
            save_path: Optional path to save the trained model

        Returns:
            Dictionary with training statistics
        """
        training_dir = Path(training_dir)
        if not training_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {training_dir}")

        # Map directory names to class labels
        dir_to_class = {
            "128": CLASS_LABELS["128"],
            "v2": CLASS_LABELS["V2"],
            "192": CLASS_LABELS["192"],
            "v0": CLASS_LABELS["V0"],
            "256": CLASS_LABELS["256"],
            "320": CLASS_LABELS["320"],
            "lossless": CLASS_LABELS["LOSSLESS"],
        }

        training_data: Dict[str, int] = {}
        audio_extensions = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus"}

        for dir_name, class_label in dir_to_class.items():
            class_dir = training_dir / dir_name
            if class_dir.exists():
                for file_path in class_dir.iterdir():
                    if file_path.suffix.lower() in audio_extensions:
                        training_data[str(file_path)] = class_label
                logger.info(f"Found {len(list(class_dir.iterdir()))} files in {dir_name}/")
            else:
                logger.warning(f"Training directory not found: {class_dir}")

        if not training_data:
            raise ValueError(f"No training files found in {training_dir}")

        return self.train(training_data, save_path)

    def _get_stated_class(self, file_format: str, stated_bitrate: Optional[int]) -> str:
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

    def _generate_warnings(
        self,
        file_format: str,
        original_format: str,
        confidence: float,
        is_transcode: bool,
        stated_bitrate: Optional[int],
    ) -> List[str]:
        """Generate warning messages based on analysis."""
        warnings = []

        # Low confidence warning
        if confidence < 0.7:
            warnings.append(f"Low confidence in detection: {confidence:.1%}")

        # Transcode warning
        if is_transcode:
            warnings.append(
                f"Lossless file ({file_format.upper()}) appears to be transcoded "
                f"from {original_format} source"
            )

        # Bitrate mismatch warning (for lossy files)
        if stated_bitrate and original_format != "LOSSLESS":
            from .constants import BITRATE_CLASSES

            detected_bitrate = None
            for _, (fmt, br) in BITRATE_CLASSES.items():
                if fmt == original_format:
                    detected_bitrate = br
                    break

            if detected_bitrate and stated_bitrate > detected_bitrate * 1.5:
                warnings.append(
                    f"Stated bitrate ({stated_bitrate} kbps) much higher than "
                    f"detected ({detected_bitrate} kbps) - possible upsampled file"
                )

        return warnings

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
