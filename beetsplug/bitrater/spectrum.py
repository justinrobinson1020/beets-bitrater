"""Spectral analysis for audio bitrate detection using PSD frequency bands.

Based on D'Alessandro & Shi paper methodology, extended for lossless detection.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
from scipy import signal
import logging

from beetsplug.bitrater.training_data.feature_cache import FeatureCache
from .types import SpectralFeatures
from .constants import SPECTRAL_PARAMS, MINIMUM_SAMPLE_RATE, MINIMUM_DURATION

logger = logging.getLogger(__name__)


class SpectrumAnalyzer:
    """
    Analyzes audio quality through frequency spectrum analysis.

    Uses Power Spectral Density (PSD) analysis in the 16-22 kHz range:
    - Bands 0-99: 16-20 kHz (paper's bitrate detection range)
    - Bands 100-149: 20-22 kHz (ultrasonic for lossless detection)
    """

    @staticmethod
    def _get_default_cache_dir() -> Path:
        """Get the default cache directory path within the training_data directory."""
        current_dir = Path(__file__).parent
        cache_dir = current_dir / "training_data" / "cache"
        return cache_dir

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize analyzer with optional caching.

        Args:
            cache_dir: Optional custom cache directory path. If None,
                      uses default path in training_data/cache.
        """
        self.fft_size = SPECTRAL_PARAMS["fft_size"]
        self.min_freq = SPECTRAL_PARAMS["min_freq"]
        self.max_freq = SPECTRAL_PARAMS["max_freq"]
        self.num_bands = SPECTRAL_PARAMS["num_bands"]

        # Use default cache directory if none provided
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()

        # Initialize feature cache
        self.cache = FeatureCache(cache_dir)
        logger.debug(f"Using feature cache directory: {cache_dir}")

        self._band_frequencies = self._calculate_band_frequencies()

    def _calculate_band_frequencies(self) -> List[Tuple[float, float]]:
        """Calculate frequency band boundaries for 150 bands across 16-22 kHz."""
        band_width = (self.max_freq - self.min_freq) / self.num_bands
        bands = []
        for i in range(self.num_bands):
            start_freq = self.min_freq + (i * band_width)
            end_freq = start_freq + band_width
            bands.append((float(start_freq), float(end_freq)))
        return bands

    def analyze_file(self, file_path: str) -> Optional[SpectralFeatures]:
        """
        Extract spectral features from an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            SpectralFeatures with 150 PSD band values, or None if analysis fails
        """
        path = Path(file_path)

        try:
            # Check cache first
            if self.cache is not None:
                cached_result = self.cache.get_features(path)
                if cached_result is not None:
                    features, metadata = cached_result
                    # Validate cached features match current config
                    if (
                        metadata.get("n_bands") == self.num_bands
                        and metadata.get("approach") == "psd_150_bands"
                    ):
                        return SpectralFeatures(
                            features=features,
                            frequency_bands=metadata.get(
                                "band_frequencies", self._band_frequencies
                            ),
                        )
                    # Cache miss due to config change - recompute

            # Load audio as mono
            y, sr = librosa.load(file_path, sr=None, mono=True)

            if not self._validate_audio(y, sr):
                logger.error(f"Invalid audio file: {file_path}")
                return None

            # Compute power spectral density using Welch's method
            # This matches the paper's approach of analyzing the entire song
            freqs, psd = signal.welch(
                y, sr, nperseg=self.fft_size, window="hann", detrend="constant"
            )

            # Extract 150 frequency band features (16-22 kHz)
            band_features = self._extract_band_features(psd, freqs)
            if band_features is None:
                return None

            # Create metadata for caching
            metadata = {
                "sample_rate": sr,
                "n_bands": self.num_bands,
                "band_frequencies": self._band_frequencies,
                "creation_date": datetime.now().isoformat(),
                "approach": "psd_150_bands",
            }

            # Cache the features
            if self.cache is not None:
                self.cache.save_features(path, band_features, metadata)

            return SpectralFeatures(
                features=band_features,
                frequency_bands=self._band_frequencies.copy(),
            )

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return None

    def _extract_band_features(
        self, psd: np.ndarray, freqs: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract PSD features for 150 frequency bands (16-22 kHz).

        Based on D'Alessandro & Shi paper methodology:
        - Divide frequency range into bands
        - Calculate average PSD for each band
        - Normalize using log scale

        Args:
            psd: Power spectral density array from Welch's method
            freqs: Corresponding frequency array

        Returns:
            Array of 150 normalized PSD band features, or None if insufficient resolution
        """
        try:
            # Filter to our frequency range (16-22 kHz)
            freq_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
            freqs_filtered = freqs[freq_mask]
            psd_filtered = psd[freq_mask]

            if len(freqs_filtered) < self.num_bands:
                logger.warning(
                    f"Insufficient frequency resolution: {len(freqs_filtered)} points "
                    f"for {self.num_bands} bands"
                )
                return None

            # Calculate band width (~40 Hz per band)
            band_width = (self.max_freq - self.min_freq) / self.num_bands
            band_features = np.zeros(self.num_bands, dtype=np.float32)

            for i in range(self.num_bands):
                start_freq = self.min_freq + (i * band_width)
                end_freq = start_freq + band_width

                # Find PSD values in this frequency band
                band_mask = (freqs_filtered >= start_freq) & (freqs_filtered < end_freq)

                if np.any(band_mask):
                    # Average power spectral density for this band
                    band_features[i] = np.mean(psd_filtered[band_mask])
                else:
                    band_features[i] = 0.0

            # Normalize using log scale (handles wide dynamic range)
            # Add small constant to avoid log(0)
            band_features = np.log10(band_features + 1e-15)

            # Normalize to 0-1 range
            min_val = np.min(band_features)
            max_val = np.max(band_features)
            if max_val > min_val:
                band_features = (band_features - min_val) / (max_val - min_val)

            return band_features

        except Exception as e:
            logger.error(f"Error in PSD band extraction: {str(e)}")
            return None

    def _validate_audio(self, y: np.ndarray, sr: int) -> bool:
        """Validate audio data meets requirements for analysis."""
        if len(y) == 0:
            logger.warning("Empty audio data")
            return False

        duration = len(y) / sr
        if duration < MINIMUM_DURATION:
            logger.warning(f"Audio too short: {duration:.2f}s < {MINIMUM_DURATION}s")
            return False

        if sr < MINIMUM_SAMPLE_RATE:
            logger.warning(
                f"Sample rate too low: {sr}Hz < {MINIMUM_SAMPLE_RATE}Hz. "
                "Cannot analyze frequencies up to 22 kHz."
            )
            return False

        return True

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Feature cache cleared")
