"""Type definitions for the bitrater plugin."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class SpectralFeatures:
    """
    Spectral features extracted from audio file.

    150 PSD bands covering 16-22 kHz plus encoder-agnostic extras:
    - Bands 0-99: 16-20 kHz (paper's bitrate detection range)
    - Bands 100-149: 20-22 kHz (ultrasonic for lossless detection)
    - 6 cutoff features, 8 temporal features, 6 artifact features
    - 6 SFB21 features (ultra_ratio, continuity, flatness, flat_std, flat_iqr, flat_19_20k)
    - 4 rolloff features (slope, total_drop, ratio_early, ratio_late)
    - is_vbr metadata flag for VBR/CBR discrimination
    """
    features: np.ndarray  # Shape: (150,) - avg PSD per frequency band
    frequency_bands: list[tuple[float, float]]  # (start_freq, end_freq) pairs
    cutoff_features: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    temporal_features: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    artifact_features: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    sfb21_features: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    rolloff_features: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    is_vbr: float = 0.0  # 1.0 if VBR, 0.0 if CBR/ABR/unknown (from file metadata)

    def as_vector(self) -> np.ndarray:
        """Flatten all features into a single vector for the classifier."""
        base = [
            np.asarray(self.features, dtype=np.float32).flatten(),
            np.asarray(self.cutoff_features, dtype=np.float32).flatten(),
            np.asarray(self.temporal_features, dtype=np.float32).flatten(),
            np.asarray(self.artifact_features, dtype=np.float32).flatten(),
            np.asarray(self.sfb21_features, dtype=np.float32).flatten(),
            np.asarray(self.rolloff_features, dtype=np.float32).flatten(),
            np.array([self.is_vbr], dtype=np.float32),
        ]
        return np.concatenate(base)


@dataclass
class ClassifierPrediction:
    """Results from the SVM classifier's prediction."""
    format_type: str  # "128", "192", "256", "320", "V0", "LOSSLESS"
    estimated_bitrate: int  # 128, 192, 256, 320, 245 (V0), or 1411 (lossless)
    confidence: float  # Confidence in prediction (0-1)
    probabilities: dict[int, float] = field(default_factory=dict)  # Class probabilities


@dataclass
class FileMetadata:
    """Audio file metadata."""
    format: str  # mp3, flac, wav, etc.
    sample_rate: int
    duration: float
    channels: int  # Number of audio channels
    encoding_type: str  # CBR, VBR, ABR, lossless
    encoder: str
    encoder_version: str | None = None
    bitrate: int | None = None  # kbps for lossy, None for lossless
    bits_per_sample: int | None = None  # For lossless formats
    filesize: int | None = None  # File size in bytes
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result of analyzing an audio file."""
    # Core identification
    filename: str
    file_format: str  # Actual container: "mp3", "flac", "wav"

    # Classification results
    original_format: str  # "128", "V2", "192", "V0", "256", "320", "LOSSLESS"
    original_bitrate: int  # 128, 190, 192, 245, 256, 320, or 1411
    confidence: float  # Final confidence after penalties

    # Transcode detection
    is_transcode: bool  # True if stated_rank > detected_rank
    stated_class: str  # What file claims to be: "320", "LOSSLESS", etc.
    detected_cutoff: int  # Detected cutoff frequency in Hz
    quality_gap: int  # Difference in quality ranks (0-6)
    transcoded_from: str | None = None  # e.g., "128" if transcoded

    # Metadata comparison
    stated_bitrate: int | None = None  # What the file metadata claims

    # Analysis metadata
    analysis_version: str = "4.0"  # Updated version
    analysis_date: datetime = field(default_factory=datetime.now)
    warnings: list[str] = field(default_factory=list)

    def summarize(self) -> dict[str, Any]:
        """Create a summary of key findings."""
        return {
            "filename": self.filename,
            "file_format": self.file_format,
            "original_format": self.original_format,
            "original_bitrate": self.original_bitrate,
            "confidence": self.confidence,
            "is_transcode": self.is_transcode,
            "stated_class": self.stated_class,
            "detected_cutoff": self.detected_cutoff,
            "quality_gap": self.quality_gap,
            "transcoded_from": self.transcoded_from,
            "stated_bitrate": self.stated_bitrate,
            "warnings": self.warnings,
            "analysis_date": self.analysis_date.isoformat(),
        }
