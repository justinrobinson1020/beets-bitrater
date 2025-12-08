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

    def _coarse_scan(self, psd: np.ndarray, freqs: np.ndarray) -> int:
        """
        Perform coarse scan to find approximate cutoff region.

        Uses sliding window band ratio comparison at 1 kHz intervals.

        Args:
            psd: Power spectral density array
            freqs: Corresponding frequency array (Hz)

        Returns:
            Candidate cutoff frequency (Hz)
        """
        best_ratio = 1.0
        best_freq = self.max_freq

        for candidate_freq in range(self.min_freq, self.max_freq, self.coarse_step):
            # Window below candidate
            below_mask = (freqs >= candidate_freq - self.window_size) & (freqs < candidate_freq)
            # Window above candidate
            above_mask = (freqs >= candidate_freq) & (freqs < candidate_freq + self.window_size)

            if not np.any(below_mask) or not np.any(above_mask):
                continue

            energy_below = np.mean(psd[below_mask])
            energy_above = np.mean(psd[above_mask])

            # Avoid division by zero
            if energy_below < 1e-10:
                continue

            ratio = energy_above / energy_below

            # Find where ratio drops most dramatically
            if ratio < best_ratio:
                best_ratio = ratio
                best_freq = candidate_freq

        return best_freq

    def _fine_scan(self, psd: np.ndarray, freqs: np.ndarray, coarse_estimate: int) -> int:
        """
        Refine cutoff estimate with fine-grained scan.

        Scans ±500 Hz around coarse estimate at 100 Hz intervals.

        Args:
            psd: Power spectral density array
            freqs: Corresponding frequency array (Hz)
            coarse_estimate: Result from coarse scan (Hz)

        Returns:
            Refined cutoff frequency (Hz)
        """
        search_start = max(self.min_freq, coarse_estimate - 500)
        search_end = min(self.max_freq, coarse_estimate + 500)

        best_ratio = 1.0
        best_freq = coarse_estimate

        for candidate_freq in range(search_start, search_end, self.fine_step):
            below_mask = (freqs >= candidate_freq - self.window_size) & (freqs < candidate_freq)
            above_mask = (freqs >= candidate_freq) & (freqs < candidate_freq + self.window_size)

            if not np.any(below_mask) or not np.any(above_mask):
                continue

            energy_below = np.mean(psd[below_mask])
            energy_above = np.mean(psd[above_mask])

            if energy_below < 1e-10:
                continue

            ratio = energy_above / energy_below

            if ratio < best_ratio:
                best_ratio = ratio
                best_freq = candidate_freq

        return best_freq
