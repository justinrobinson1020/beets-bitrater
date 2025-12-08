"""Audio quality analyzer - orchestrates spectral analysis and classification."""

from pathlib import Path
from typing import Optional, Dict, List
import logging

from .spectrum import SpectrumAnalyzer
from .classifier import QualityClassifier
from .file_analyzer import FileAnalyzer
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

    def analyze_file(self, file_path: str) -> Optional[AnalysisResult]:
        """
        Analyze a single audio file.

        Performs spectral analysis, classification, and transcode detection.

        Args:
            file_path: Path to the audio file

        Returns:
            AnalysisResult with classification and transcode detection, or None if analysis fails
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

        # 3. Classify (if model is trained)
        if not self.classifier.trained:
            logger.warning("Classifier not trained - returning features without classification")
            return AnalysisResult(
                filename=str(path),
                file_format=path.suffix.lower().lstrip("."),
                original_format="UNKNOWN",
                original_bitrate=0,
                confidence=0.0,
                is_transcode=False,
                stated_bitrate=metadata.bitrate if metadata else None,
                warnings=["Classifier not trained"],
            )

        prediction = self.classifier.predict(features)

        # 4. Determine file format and detect transcoding
        file_format = path.suffix.lower().lstrip(".")
        is_lossless_container = file_format in LOSSLESS_CONTAINERS
        detected_lossy = prediction.format_type != "LOSSLESS"

        # Transcode detection: lossless container but lossy content detected
        is_transcode = is_lossless_container and detected_lossy
        transcoded_from = prediction.format_type if is_transcode else None

        # 5. Generate warnings
        warnings = self._generate_warnings(
            file_format=file_format,
            original_format=prediction.format_type,
            confidence=prediction.confidence,
            is_transcode=is_transcode,
            stated_bitrate=metadata.bitrate if metadata else None,
        )

        return AnalysisResult(
            filename=str(path),
            file_format=file_format,
            original_format=prediction.format_type,
            original_bitrate=prediction.estimated_bitrate,
            confidence=prediction.confidence,
            is_transcode=is_transcode,
            transcoded_from=transcoded_from,
            stated_bitrate=metadata.bitrate if metadata else None,
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
            ├── 192/      # Files encoded at 192 kbps (class 1)
            ├── 256/      # Files encoded at 256 kbps (class 2)
            ├── 320/      # Files encoded at 320 kbps (class 3)
            ├── v0/       # VBR-0 files (class 4)
            └── lossless/ # Lossless files (class 5)

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
            "192": CLASS_LABELS["192"],
            "256": CLASS_LABELS["256"],
            "320": CLASS_LABELS["320"],
            "v0": CLASS_LABELS["V0"],
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
