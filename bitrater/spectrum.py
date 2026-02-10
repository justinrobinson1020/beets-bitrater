"""Spectral analysis for audio bitrate detection using PSD frequency bands.

Based on D'Alessandro & Shi paper methodology, extended for lossless detection.
"""

import logging
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
from scipy import signal, stats

from .feature_cache import FeatureCache

from .constants import MINIMUM_DURATION, MINIMUM_SAMPLE_RATE, SPECTRAL_PARAMS
from .types import SpectralFeatures

logger = logging.getLogger("beets.bitrater")


class SpectrumAnalyzer:
    """
    Analyzes audio quality through frequency spectrum analysis.

    Uses Power Spectral Density (PSD) analysis in the 16-22 kHz range:
    - Bands 0-99: 16-20 kHz (paper's bitrate detection range)
    - Bands 100-149: 20-22 kHz (ultrasonic for lossless detection)
    """

    @staticmethod
    def _get_default_cache_dir() -> Path:
        """Get the default cache directory (~/.cache/bitrater/features)."""
        return Path.home() / ".cache" / "bitrater" / "features"

    def __init__(self, cache_dir: Path | None = None):
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

        # In-memory PSD cache to avoid double-load between analyze_file and get_psd
        self._last_psd_path: str | None = None
        self._last_psd: tuple[np.ndarray, np.ndarray] | None = None

    def _calculate_band_frequencies(self) -> list[tuple[float, float]]:
        """Calculate frequency band boundaries for 150 bands across 16-22 kHz."""
        band_width = (self.max_freq - self.min_freq) / self.num_bands
        bands = []
        for i in range(self.num_bands):
            start_freq = self.min_freq + (i * band_width)
            end_freq = start_freq + band_width
            bands.append((float(start_freq), float(end_freq)))
        return bands

    def analyze_file(self, file_path: str, is_vbr: float = 0.0) -> SpectralFeatures | None:
        """
        Extract spectral features from an audio file.

        Args:
            file_path: Path to the audio file
            is_vbr: VBR flag from file metadata (1.0 if VBR, 0.0 if CBR/unknown)

        Returns:
            SpectralFeatures with encoder-agnostic feature set, or None if analysis fails
        """
        path = Path(file_path)

        # Clear in-memory PSD cache for new file
        self._last_psd_path = None
        self._last_psd = None

        try:
            # Check cache first
            cached_result = self.cache.get_features(path)
            if cached_result is not None:
                features, metadata = cached_result
                # Validate cached features match current config
                if (
                    metadata.get("n_bands") == self.num_bands
                    and metadata.get("approach") == "encoder_agnostic_v8"
                ):
                    psd_bands, cutoff_feats, temporal_feats, artifact_feats, sfb21_feats, rolloff_feats = (
                        self._split_feature_vector(features, metadata)
                    )
                    return SpectralFeatures(
                        features=psd_bands,
                        frequency_bands=metadata.get(
                            "band_frequencies", self._band_frequencies
                        ),
                        cutoff_features=cutoff_feats,
                        temporal_features=temporal_feats,
                        artifact_features=artifact_feats,
                        sfb21_features=sfb21_feats,
                        rolloff_features=rolloff_feats,
                        is_vbr=is_vbr,
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

            # Cache PSD for subsequent get_psd() call (avoids double-load)
            self._last_psd_path = file_path
            self._last_psd = (psd, freqs)

            # Extract 150 frequency band features (16-22 kHz)
            band_features = self._extract_band_features(psd, freqs)
            if band_features is None:
                return None

            # Extract encoder-agnostic extras
            cutoff_features = self._extract_cutoff_features(psd, freqs)
            temporal_features = self._extract_temporal_features(y, sr)
            artifact_features = self._extract_artifact_features(
                psd, freqs, cutoff_features
            )
            sfb21_features = self._extract_sfb21_features(y, sr)
            rolloff_features = self._extract_rolloff_features(y, sr)

            # Flatten for caching (exclude is_vbr because it comes from metadata)
            combined_features = np.concatenate(
                [
                    band_features.astype(np.float32),
                    cutoff_features.astype(np.float32),
                    temporal_features.astype(np.float32),
                    artifact_features.astype(np.float32),
                    sfb21_features.astype(np.float32),
                    rolloff_features.astype(np.float32),
                ]
            )

            # Create metadata for caching
            metadata = {
                "sample_rate": sr,
                "n_bands": self.num_bands,
                "band_frequencies": self._band_frequencies,
                "creation_date": datetime.now().isoformat(),
                "approach": "encoder_agnostic_v8",
                "cutoff_len": len(cutoff_features),
                "temporal_len": len(temporal_features),
                "artifact_len": len(artifact_features),
                "sfb21_len": len(sfb21_features),
                "rolloff_len": len(rolloff_features),
            }

            # Cache the features
            self.cache.save_features(path, combined_features, metadata)

            return SpectralFeatures(
                features=band_features,
                frequency_bands=self._band_frequencies.copy(),
                cutoff_features=cutoff_features,
                temporal_features=temporal_features,
                artifact_features=artifact_features,
                sfb21_features=sfb21_features,
                rolloff_features=rolloff_features,
                is_vbr=is_vbr,
            )

        except FileNotFoundError:
            # File doesn't exist - expected in some cases
            return None
        except (ValueError, RuntimeError) as e:
            # Audio format errors or analysis failures
            logger.warning(f"Failed to analyze file {file_path}: {e}")
            return None
        except Exception as e:
            # Unexpected errors - log for investigation
            logger.error(
                f"Unexpected error analyzing file {file_path}: {e}", exc_info=True
            )
            return None

    def _extract_band_features(
        self, psd: np.ndarray, freqs: np.ndarray
    ) -> np.ndarray | None:
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

        except ValueError as e:
            # Expected errors: insufficient data, invalid values
            logger.debug(f"Error in PSD band extraction: {e}")
            return None
        except Exception as e:
            # Unexpected errors - log for investigation
            logger.error(
                f"Unexpected error in PSD band extraction: {e}", exc_info=True
            )
            return None

    def _estimate_cutoff_normalized(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Estimate normalized cutoff frequency in the analysis band."""
        try:
            band_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
            freqs_band = freqs[band_mask]
            psd_band = psd[band_mask]
            if len(freqs_band) == 0:
                return 0.0

            psd_db = 10 * np.log10(psd_band + 1e-12)
            peak = float(psd_db.max())
            threshold = peak - 20.0
            above = np.where(psd_db > threshold)[0]
            if len(above) == 0:
                cutoff_freq = freqs_band[0]
            else:
                cutoff_freq = freqs_band[int(above[-1])]

            return float((cutoff_freq - self.min_freq) / (self.max_freq - self.min_freq))
        except ValueError as e:
            # Expected error: invalid array values
            logger.debug(f"Error estimating cutoff: {e}")
            return 0.0
        except Exception as e:
            # Unexpected errors - log for investigation
            logger.error(f"Unexpected error estimating cutoff: {e}", exc_info=True)
            return 0.0

    def _extract_cutoff_features(self, psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Encoder-agnostic cutoff descriptors (length 6)."""
        band_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        freqs_band = freqs[band_mask]
        psd_band = psd[band_mask]

        if len(freqs_band) == 0:
            return np.zeros(6, dtype=np.float32)

        psd_db = 10 * np.log10(psd_band + 1e-12)
        peak = float(psd_db.max())
        norm = psd_db - peak  # normalize so max is 0 dB

        gradient = np.gradient(norm)
        transition_indices = np.where(gradient < -0.05)[0]

        primary_cutoff = self._estimate_cutoff_normalized(freqs, psd)
        num_transitions = float(len(transition_indices))

        if len(transition_indices) > 0:
            first_idx = int(transition_indices[0])
            first_transition_freq = (freqs_band[first_idx] - self.min_freq) / (
                self.max_freq - self.min_freq
            )
            first_transition_mag = float(abs(gradient[first_idx]))
        else:
            first_transition_freq = 0.0
            first_transition_mag = 0.0

        cutoff_gradient = float(abs(gradient.min())) if len(gradient) else 0.0
        transition_gap = max(0.0, primary_cutoff - first_transition_freq)

        return np.array(
            [
                primary_cutoff,
                num_transitions,
                first_transition_freq,
                first_transition_mag,
                cutoff_gradient,
                transition_gap,
            ],
            dtype=np.float32,
        )

    def _extract_temporal_features(self, y: np.ndarray, sr: int, n_windows: int = 8) -> np.ndarray:
        """Temporal descriptors to highlight VBR variability (length 8).

        Note: Reduced from 16 to 8 windows for performance (~50% fewer Welch calls)
        while maintaining sufficient temporal resolution for VBR detection.
        """
        if n_windows <= 0:
            return np.zeros(8, dtype=np.float32)

        segments = np.array_split(y, n_windows)
        cutoff_vals: list[float] = []
        hf_energy: list[float] = []
        window_energy: list[float] = []

        for segment in segments:
            if len(segment) == 0:
                continue
            freqs, psd = signal.welch(segment, sr, nperseg=self.fft_size)
            cutoff_vals.append(self._estimate_cutoff_normalized(freqs, psd))

            band_mask = (freqs >= self.min_freq) & (freqs <= 20000)
            hf_vals = psd[band_mask]
            hf_energy.append(float(hf_vals.mean()) if len(hf_vals) else 0.0)
            window_energy.append(float(psd.mean()) if len(psd) else 0.0)

        if not cutoff_vals:
            return np.zeros(8, dtype=np.float32)

        cutoff_arr = np.asarray(cutoff_vals, dtype=np.float32)
        hf_arr = np.asarray(hf_energy, dtype=np.float32)
        energy_arr = np.asarray(window_energy, dtype=np.float32)

        cutoff_variance = float(np.var(cutoff_arr))
        cutoff_min = float(cutoff_arr.min())
        cutoff_max = float(cutoff_arr.max())
        cutoff_mean = float(cutoff_arr.mean()) or 1e-6
        cutoff_stability_ratio = float((cutoff_max - cutoff_min) / cutoff_mean)

        hf_energy_variance = float(np.var(hf_arr))
        if len(hf_arr) > 1:
            hf_energy_trend = float(np.polyfit(np.arange(len(hf_arr)), hf_arr, 1)[0])
        else:
            hf_energy_trend = 0.0

        if len(cutoff_arr) > 1:
            diffs = np.diff(cutoff_arr)
            temporal_consistency = float(np.mean(np.abs(diffs)))
        else:
            temporal_consistency = 0.0

        frame_energy_variance = float(np.var(energy_arr))

        return np.array(
            [
                cutoff_variance,
                cutoff_min,
                cutoff_max,
                cutoff_stability_ratio,
                hf_energy_variance,
                hf_energy_trend,
                temporal_consistency,
                frame_energy_variance,
            ],
            dtype=np.float32,
        )

    def _spectral_flatness(self, values: np.ndarray) -> float:
        """Compute spectral flatness (handles dB inputs without log warnings)."""
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0

        # Convert from dB to linear power so values are positive before log
        linear = np.power(10.0, values / 10.0)
        linear = np.clip(linear, 1e-12, None)

        geometric_mean = np.exp(np.mean(np.log(linear)))
        arithmetic_mean = np.mean(linear)
        if arithmetic_mean <= 0:
            return 0.0
        return float(geometric_mean / arithmetic_mean)

    def _extract_artifact_features(
        self, psd: np.ndarray, freqs: np.ndarray, cutoff_features: np.ndarray
    ) -> np.ndarray:
        """Artifact descriptors for transcode detection (length 6)."""
        band_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        freqs_band = freqs[band_mask]
        psd_band = psd[band_mask]

        if len(freqs_band) == 0:
            return np.zeros(6, dtype=np.float32)

        cutoff_norm = float(cutoff_features[0]) if len(cutoff_features) > 0 else 0.0
        cutoff_freq = (cutoff_norm * (self.max_freq - self.min_freq)) + self.min_freq

        psd_db = 10 * np.log10(psd_band + 1e-12)
        above_mask = freqs_band > cutoff_freq
        below_mask = freqs_band <= cutoff_freq

        above = psd_db[above_mask]
        below = psd_db[below_mask]

        if len(above) == 0:
            above = np.array([0.0])
        if len(below) == 0:
            below = np.array([0.0])

        threshold = max(float(above.max()), float(below.max())) - 25.0
        sparsity = 1.0 - (float(np.sum(above > threshold)) / float(len(above)))

        noise_ratio = self._spectral_flatness(above)
        spectral_flatness_hf = self._spectral_flatness(psd_db[freqs_band >= 18000])

        spectral_discontinuity = float(np.std(np.diff(psd_db))) if len(psd_db) > 1 else 0.0
        harmonic_residual = float(1.0 - self._spectral_flatness(below))

        # Measure deviation from a smoothed spectrum as proxy for interpolation artifacts
        try:
            smoothed = signal.medfilt(psd_db, kernel_size=5)
            interpolation_score = float(np.mean(np.abs(psd_db - smoothed)))
        except ValueError as e:
            # Expected error: invalid kernel size for array
            logger.debug(f"Could not compute interpolation score: {e}")
            interpolation_score = 0.0
        except Exception as e:
            # Unexpected errors - log for investigation
            logger.error(
                f"Unexpected error computing interpolation score: {e}", exc_info=True
            )
            interpolation_score = 0.0

        return np.array(
            [
                sparsity,
                noise_ratio,
                spectral_flatness_hf,
                spectral_discontinuity,
                harmonic_residual,
                interpolation_score,
            ],
            dtype=np.float32,
        )

    def _extract_sfb21_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """SFB21 features for V0 vs LOSSLESS discrimination (length 6).

        Features:
        1. sfb21_ultra_ratio: energy(19.5-22kHz) / energy(16-19.5kHz)
           - V0 drops off above 19.5kHz, LOSSLESS doesn't
        2. sfb21_continuity: energy(16-19kHz) / energy(14-16kHz)
           - Measures smoothness of transition into sfb21 band
        3. sfb21_flatness: mean spectral flatness in 16-19.5kHz band
        4. sfb21_flat_std: temporal std of per-frame flatness (V0 higher, -76%)
        5. sfb21_flat_iqr: IQR of per-frame flatness (V0 higher, -82%)
        6. flat_19_20k: flatness in 19-20kHz sub-band (LL higher, +57%)
        """
        if len(y) == 0:
            return np.zeros(6, dtype=np.float32)

        # Use STFT for frequency analysis
        n_fft = 4096
        hop = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Band masks
        below_sfb21 = (freqs >= 14000) & (freqs < 16000)  # Just below sfb21
        sfb21_band = (freqs >= 16000) & (freqs < 19500)   # sfb21 range
        ultra_band = (freqs >= 19500) & (freqs < 22000)   # Above V0 cutoff
        band_19_20k = (freqs >= 19000) & (freqs < 20000)  # Sub-band for flat_19_20k

        # Energy in each band (mean across time and frequency)
        below_energy = np.mean(S[below_sfb21, :]) if np.any(below_sfb21) else 1e-10
        sfb21_energy = np.mean(S[sfb21_band, :]) if np.any(sfb21_band) else 1e-10
        ultra_energy = np.mean(S[ultra_band, :]) if np.any(ultra_band) else 0.0

        # Feature 1: Ultra ratio
        sfb21_ultra_ratio = float(ultra_energy / (sfb21_energy + 1e-10))

        # Feature 2: Continuity across 16kHz boundary
        sfb21_continuity = float(sfb21_energy / (below_energy + 1e-10))

        # Helper: compute per-frame flatness for a frequency band
        def band_flatness_per_frame(S: np.ndarray, mask: np.ndarray) -> np.ndarray:
            """Compute Wiener entropy (spectral flatness) per frame."""
            band = S[mask, :]
            if band.size == 0:
                return np.array([0.0])
            # Geometric mean / arithmetic mean per frame
            geo = np.exp(np.mean(np.log(band + 1e-10), axis=0))
            arith = np.mean(band, axis=0)
            return geo / (arith + 1e-10)

        # Per-frame flatness in sfb21 band
        flat_sfb21_frames = band_flatness_per_frame(S, sfb21_band)

        # Feature 3: Mean flatness (original)
        sfb21_flatness = float(np.mean(flat_sfb21_frames))

        # Feature 4: Temporal std of flatness (V0 higher - more variance)
        sfb21_flat_std = float(np.std(flat_sfb21_frames))

        # Feature 5: IQR of flatness (V0 higher - more variance)
        sfb21_flat_iqr = float(
            np.percentile(flat_sfb21_frames, 75) - np.percentile(flat_sfb21_frames, 25)
        )

        # Feature 6: Flatness in 19-20kHz sub-band (LL higher)
        flat_19_20k_frames = band_flatness_per_frame(S, band_19_20k)
        flat_19_20k = float(np.mean(flat_19_20k_frames))

        return np.array([
            sfb21_ultra_ratio,
            sfb21_continuity,
            sfb21_flatness,
            sfb21_flat_std,
            sfb21_flat_iqr,
            flat_19_20k,
        ], dtype=np.float32)

    def _extract_rolloff_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Rolloff curve shape features between 18-21kHz (length 4).

        Analyzes the shape of spectral energy decline in the rolloff region.
        V0's encoder creates a characteristic steep rolloff around 19.5kHz.
        FLAC's rolloff depends on source material - typically smoother/shallower.

        Features:
        1. rolloff_slope: Linear slope in dB/kHz - V0 steeper (more negative)
        2. rolloff_total_drop: Total dB drop across 18-21kHz - V0 drops more
        3. rolloff_ratio_early: 19-20kHz vs 18-19kHz energy - V0 higher
        4. rolloff_ratio_late: 20-21kHz vs 19-20kHz energy - LOSSLESS higher
        """
        if len(y) == 0:
            return np.zeros(4, dtype=np.float32)

        n_fft = 8192  # High resolution for precise frequency analysis

        # Compute power spectrum (averaged across time)
        S = np.abs(librosa.stft(y, n_fft=n_fft)) ** 2
        power_spectrum = np.mean(S, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Rolloff region 18-21kHz
        rolloff_mask = (freqs >= 18000) & (freqs <= 21000)
        rolloff_freqs = freqs[rolloff_mask]
        rolloff_power = power_spectrum[rolloff_mask]

        if len(rolloff_freqs) == 0 or len(rolloff_power) == 0:
            return np.zeros(4, dtype=np.float32)

        # Convert to dB and normalize to start at 0 dB
        rolloff_db = 10 * np.log10(rolloff_power + 1e-10)
        rolloff_db_norm = rolloff_db - rolloff_db[0]

        # Feature 1: Slope (dB per kHz) - LL shallower, V0 steeper
        slope, _, _, _, _ = stats.linregress(rolloff_freqs, rolloff_db_norm)
        rolloff_slope = float(slope * 1000)  # Scale for readability

        # Feature 2: Total drop in dB - LL less drop, V0 more drop
        rolloff_total_drop = float(rolloff_db_norm[-1] - rolloff_db_norm[0])

        # Features 3 & 4: Band energy ratios
        band1_mask = (freqs >= 18000) & (freqs < 19000)
        band2_mask = (freqs >= 19000) & (freqs < 20000)
        band3_mask = (freqs >= 20000) & (freqs <= 21000)

        band1_energy = np.mean(power_spectrum[band1_mask]) if np.any(band1_mask) else 1e-10
        band2_energy = np.mean(power_spectrum[band2_mask]) if np.any(band2_mask) else 1e-10
        band3_energy = np.mean(power_spectrum[band3_mask]) if np.any(band3_mask) else 1e-10

        # ratio_early: 19-20kHz vs 18-19kHz - V0 higher (steeper early drop)
        rolloff_ratio_early = float(band2_energy / (band1_energy + 1e-10))

        # ratio_late: 20-21kHz vs 19-20kHz - LOSSLESS higher
        rolloff_ratio_late = float(band3_energy / (band2_energy + 1e-10))

        return np.array([
            rolloff_slope,
            rolloff_total_drop,
            rolloff_ratio_early,
            rolloff_ratio_late,
        ], dtype=np.float32)

    def _split_feature_vector(
        self, vector: np.ndarray, metadata: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split cached feature vector into component arrays."""
        psd_len = metadata.get("n_bands", self.num_bands)
        cutoff_len = metadata.get("cutoff_len", 6)
        temporal_len = metadata.get("temporal_len", 8)
        artifact_len = metadata.get("artifact_len", 6)
        sfb21_len = metadata.get("sfb21_len", 6)
        rolloff_len = metadata.get("rolloff_len", 4)

        psd_end = psd_len
        cutoff_end = psd_end + cutoff_len
        temporal_end = cutoff_end + temporal_len
        artifact_end = temporal_end + artifact_len
        sfb21_end = artifact_end + sfb21_len
        rolloff_end = sfb21_end + rolloff_len

        psd_bands = vector[:psd_end]
        cutoff_feats = vector[psd_end:cutoff_end] if cutoff_len else np.zeros(6, dtype=np.float32)
        temporal_feats = (
            vector[cutoff_end:temporal_end] if temporal_len else np.zeros(8, dtype=np.float32)
        )
        artifact_feats = (
            vector[temporal_end:artifact_end] if artifact_len else np.zeros(6, dtype=np.float32)
        )
        sfb21_feats = (
            vector[artifact_end:sfb21_end] if sfb21_len else np.zeros(6, dtype=np.float32)
        )
        rolloff_feats = (
            vector[sfb21_end:rolloff_end] if rolloff_len else np.zeros(4, dtype=np.float32)
        )
        return (
            np.asarray(psd_bands, dtype=np.float32),
            np.asarray(cutoff_feats, dtype=np.float32),
            np.asarray(temporal_feats, dtype=np.float32),
            np.asarray(artifact_feats, dtype=np.float32),
            np.asarray(sfb21_feats, dtype=np.float32),
            np.asarray(rolloff_feats, dtype=np.float32),
        )

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

    def get_psd(self, file_path: str) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Get raw PSD data for cutoff detection.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (psd, freqs) arrays, or None if analysis fails
        """
        # Return cached PSD if available (avoids reloading after analyze_file)
        if self._last_psd_path == file_path and self._last_psd is not None:
            return self._last_psd

        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
        except FileNotFoundError:
            # File doesn't exist - expected in some cases
            return None
        except (ValueError, RuntimeError) as e:
            # Audio format errors or analysis failures
            logger.warning(f"Failed to load audio: {e}")
            return None
        except Exception as e:
            # Unexpected errors - log for investigation
            logger.error(f"Unexpected error loading audio: {e}", exc_info=True)
            return None

        if not self._validate_audio(y, sr):
            return None

        # Calculate PSD
        freqs, psd = signal.welch(y, sr, nperseg=self.fft_size)

        return psd, freqs

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self.cache.clear()
        logger.info("Feature cache cleared")
