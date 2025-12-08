"""Cutoff frequency detection for transcode validation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CutoffResult:
    """Result of cutoff frequency detection."""

    cutoff_frequency: int  # Detected cutoff in Hz
    gradient: float  # Sharpness of cutoff (higher = sharper = more artificial)
    is_sharp: bool  # True if gradient indicates artificial cutoff
    confidence: float  # Confidence in detection (0.0-1.0)


class CutoffDetector:
    """
    Detects frequency cutoff using sliding window band ratio comparison.

    Uses coarse-to-fine scanning to efficiently find where high-frequency
    content ends, then measures gradient sharpness to distinguish artificial
    MP3 cutoffs from natural rolloff.
    """

    def __init__(
        self,
        min_freq: int = 15000,
        max_freq: int = 22050,
        coarse_step: int = 1000,
        fine_step: int = 100,
        window_size: int = 1000,
        sharp_threshold: float = 0.5,
    ):
        """
        Initialize cutoff detector.

        Args:
            min_freq: Start of scan range (Hz)
            max_freq: End of scan range (Hz)
            coarse_step: Step size for initial scan (Hz)
            fine_step: Step size for refinement (Hz)
            window_size: Size of comparison windows (Hz)
            sharp_threshold: Gradient threshold for "sharp" classification
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.window_size = window_size
        self.sharp_threshold = sharp_threshold
