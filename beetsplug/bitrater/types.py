"""Type definitions for the bitrater plugin."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np


@dataclass
class SpectralFeatures:
    """
    Spectral features extracted from audio file.

    150 PSD bands covering 16-22 kHz:
    - Bands 0-99: 16-20 kHz (paper's bitrate detection range)
    - Bands 100-149: 20-22 kHz (ultrasonic for lossless detection)
    """
    features: np.ndarray  # Shape: (150,) - avg PSD per frequency band
    frequency_bands: List[Tuple[float, float]]  # (start_freq, end_freq) pairs


@dataclass
class ClassifierPrediction:
    """Results from the SVM classifier's prediction."""
    format_type: str  # "128", "192", "256", "320", "V0", "LOSSLESS"
    estimated_bitrate: int  # 128, 192, 256, 320, 245 (V0), or 1411 (lossless)
    confidence: float  # Confidence in prediction (0-1)
    probabilities: Dict[int, float] = field(default_factory=dict)  # Class probabilities


@dataclass
class FileMetadata:
    """Audio file metadata."""
    format: str  # mp3, flac, wav, etc.
    sample_rate: int
    duration: float
    channels: int  # Number of audio channels
    encoding_type: str  # CBR, VBR, ABR, lossless
    encoder: str
    encoder_version: Optional[str] = None
    bitrate: Optional[int] = None  # kbps for lossy, None for lossless
    bits_per_sample: Optional[int] = None  # For lossless formats
    filesize: Optional[int] = None  # File size in bytes
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result of analyzing an audio file."""
    # Core identification
    filename: str
    file_format: str  # Actual container: "mp3", "flac", "wav"

    # Classification results
    original_format: str  # "128", "192", "256", "320", "V0", "LOSSLESS"
    original_bitrate: int  # 128, 192, 256, 320, 245 (V0 avg), or 1411 (CD)
    confidence: float  # SVM decision confidence

    # Transcode detection
    is_transcode: bool  # True if lossless container but lossy content detected
    transcoded_from: Optional[str] = None  # e.g., "128" if FLAC contains 128 kbps content

    # Metadata comparison
    stated_bitrate: Optional[int] = None  # What the file metadata claims

    # Analysis metadata
    analysis_version: str = "3.0"
    analysis_date: datetime = field(default_factory=datetime.now)
    warnings: List[str] = field(default_factory=list)

    def summarize(self) -> Dict[str, Any]:
        """Create a summary of key findings."""
        return {
            "filename": self.filename,
            "file_format": self.file_format,
            "original_format": self.original_format,
            "original_bitrate": self.original_bitrate,
            "confidence": self.confidence,
            "is_transcode": self.is_transcode,
            "transcoded_from": self.transcoded_from,
            "stated_bitrate": self.stated_bitrate,
            "warnings": self.warnings,
            "analysis_date": self.analysis_date.isoformat(),
        }
