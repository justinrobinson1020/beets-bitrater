"""Cutoff frequency detection for transcode validation."""

from dataclasses import dataclass

import numpy as np

from .constants import ENERGY_EPSILON, GRADIENT_NORMALIZATION_DB, GRADIENT_WINDOW_HZ


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
            if energy_below < ENERGY_EPSILON:
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

            if energy_below < ENERGY_EPSILON:
                continue

            ratio = energy_above / energy_below

            if ratio < best_ratio:
                best_ratio = ratio
                best_freq = candidate_freq

        return best_freq

    def _measure_gradient(self, psd: np.ndarray, freqs: np.ndarray, cutoff: int) -> float:
        """
        Measure gradient sharpness at the cutoff frequency.

        Calculates the slope of energy decline across the transition.
        Sharp gradients indicate artificial MP3 cutoffs.
        Gradual gradients suggest natural rolloff (old recordings, etc).

        Args:
            psd: Power spectral density array
            freqs: Corresponding frequency array (Hz)
            cutoff: Detected cutoff frequency (Hz)

        Returns:
            Gradient value (higher = sharper cutoff)
        """
        # Sample points 500 Hz below and above cutoff
        below_point = cutoff - 500
        above_point = cutoff + 500

        # Find energy at these points (average over small window)
        window = GRADIENT_WINDOW_HZ

        below_mask = (freqs >= below_point - window / 2) & (freqs < below_point + window / 2)
        above_mask = (freqs >= above_point - window / 2) & (freqs < above_point + window / 2)

        if not np.any(below_mask) or not np.any(above_mask):
            return 0.0

        energy_below = np.mean(psd[below_mask])
        energy_above = np.mean(psd[above_mask])

        # Avoid log of zero
        if energy_below < ENERGY_EPSILON or energy_above < ENERGY_EPSILON:
            if energy_below > energy_above:
                return 1.0  # Maximum sharpness
            return 0.0

        # Calculate gradient in dB per kHz
        db_drop = 10 * np.log10(energy_below / energy_above)
        gradient = db_drop / 1.0  # Per 1 kHz (1000 Hz span)

        # Normalize to 0-1 range
        normalized = min(1.0, max(0.0, gradient / GRADIENT_NORMALIZATION_DB))

        return normalized

    def detect(self, psd: np.ndarray, freqs: np.ndarray) -> CutoffResult:
        """
        Detect cutoff frequency using coarse-to-fine scanning.

        Args:
            psd: Power spectral density array
            freqs: Corresponding frequency array (Hz)

        Returns:
            CutoffResult with detected frequency and sharpness info
        """
        # Step 1: Coarse scan
        coarse_cutoff = self._coarse_scan(psd, freqs)

        # Step 2: Fine scan
        refined_cutoff = self._fine_scan(psd, freqs, coarse_cutoff)

        # Step 3: Measure gradient
        gradient = self._measure_gradient(psd, freqs, refined_cutoff)
        is_sharp = gradient > self.sharp_threshold

        # Step 4: Calculate confidence based on how clear the cutoff is
        # Higher gradient = clearer cutoff = higher confidence
        confidence = min(1.0, gradient + 0.3) if is_sharp else 0.5

        return CutoffResult(
            cutoff_frequency=refined_cutoff,
            gradient=gradient,
            is_sharp=is_sharp,
            confidence=confidence,
        )
