"""Spectral analysis for audio bitrate detection using PSD frequency bands.

Based on D'Alessandro & Shi paper methodology, extended for lossless detection.
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
from scipy import signal, stats

from .constants import MINIMUM_DURATION, MINIMUM_SAMPLE_RATE, SPECTRAL_PARAMS
from .feature_cache import FeatureCache
from .types import SpectralFeatures

# Suppress harmless scipy/librosa warnings about FFT window size exceeding
# short audio segments. The libraries auto-adapt the window size.
warnings.filterwarnings("ignore", message="nperseg.*is greater than input length", module="scipy")
warnings.filterwarnings("ignore", message="n_fft=.*is too large for input signal", module="librosa")

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

    def analyze_file(self, file_path: str) -> SpectralFeatures | None:
        """
        Extract spectral features from an audio file.

        Args:
            file_path: Path to the audio file

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
                    and metadata.get("approach") == "encoder_agnostic_v21"
                ):
                    (
                        psd_bands,
                        cutoff_feats,
                        sfb21_feats,
                        rolloff_feats,
                        discriminative_feats,
                        temporal_feats,
                        crossband_feats,
                        cutoff_cleanliness_feats,
                        mdct_feats,
                    ) = self._split_feature_vector(features, metadata)
                    return SpectralFeatures(
                        features=psd_bands,
                        frequency_bands=metadata.get("band_frequencies", self._band_frequencies),
                        cutoff_features=cutoff_feats,
                        sfb21_features=sfb21_feats,
                        rolloff_features=rolloff_feats,
                        discriminative_features=discriminative_feats,
                        temporal_features=temporal_feats,
                        crossband_features=crossband_feats,
                        cutoff_cleanliness_features=cutoff_cleanliness_feats,
                        mdct_features=mdct_feats,
                    )
                # Cache miss due to config change - recompute

            # Load audio as mono
            y, sr = librosa.load(file_path, sr=None, mono=True)

            if not self._validate_audio(y, sr):
                logger.error(f"Invalid audio file: {file_path}")
                return None

            # Single STFT for all feature extraction (n_fft=8192 for high resolution)
            S_complex = librosa.stft(y, n_fft=self.fft_size)
            S_mag = np.abs(S_complex)
            S_power = S_mag ** 2
            stft_freqs = librosa.fft_frequencies(sr=sr, n_fft=self.fft_size)

            # Derive Welch-like PSD from STFT (mean power across time frames)
            psd = np.mean(S_power, axis=1)
            freqs = stft_freqs

            # Cache PSD for subsequent get_psd() call (avoids double-load)
            self._last_psd_path = file_path
            self._last_psd = (psd, freqs)

            # Extract 150 frequency band features (16-22 kHz)
            band_features = self._extract_band_features(psd, freqs)
            if band_features is None:
                return None

            # Extract encoder-agnostic extras (reuse STFT — no additional spectral calls)
            cutoff_features = self._extract_cutoff_features(psd, freqs)
            sfb21_features = self._extract_sfb21_features(S_mag, stft_freqs)
            rolloff_features = self._extract_rolloff_features(S_power, stft_freqs)
            discriminative_features = self._extract_discriminative_features(
                band_features, sfb21_features
            )
            temporal_features = self._extract_temporal_features(S_mag, stft_freqs, sr)
            crossband_features = self._extract_crossband_features(S_power, stft_freqs)
            cutoff_cleanliness_features = self._extract_cutoff_cleanliness_features(
                S_power, S_mag, stft_freqs, cutoff_features
            )
            mdct_features = self._extract_mdct_features(y, sr)

            # Flatten for caching
            combined_features = np.concatenate(
                [
                    band_features.astype(np.float32),
                    cutoff_features.astype(np.float32),
                    sfb21_features.astype(np.float32),
                    rolloff_features.astype(np.float32),
                    discriminative_features.astype(np.float32),
                    temporal_features.astype(np.float32),
                    crossband_features.astype(np.float32),
                    cutoff_cleanliness_features.astype(np.float32),
                    mdct_features.astype(np.float32),
                ]
            )

            # Create metadata for caching
            metadata = {
                "sample_rate": sr,
                "n_bands": self.num_bands,
                "band_frequencies": self._band_frequencies,
                "creation_date": datetime.now().isoformat(),
                "approach": "encoder_agnostic_v21",
                "cutoff_len": len(cutoff_features),
                "sfb21_len": len(sfb21_features),
                "rolloff_len": len(rolloff_features),
                "discriminative_len": len(discriminative_features),
                "temporal_len": len(temporal_features),
                "crossband_len": len(crossband_features),
                "cutoff_cleanliness_len": len(cutoff_cleanliness_features),
                "mdct_len": len(mdct_features),
            }

            # Cache the features
            self.cache.save_features(path, combined_features, metadata)

            return SpectralFeatures(
                features=band_features,
                frequency_bands=self._band_frequencies.copy(),
                cutoff_features=cutoff_features,
                sfb21_features=sfb21_features,
                rolloff_features=rolloff_features,
                discriminative_features=discriminative_features,
                temporal_features=temporal_features,
                crossband_features=crossband_features,
                cutoff_cleanliness_features=cutoff_cleanliness_features,
                mdct_features=mdct_features,
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
            logger.error(f"Unexpected error analyzing file {file_path}: {e}", exc_info=True)
            return None

    def _extract_band_features(self, psd: np.ndarray, freqs: np.ndarray) -> np.ndarray | None:
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
            else:
                # Flat spectrum (all bands identical) — no discriminative info
                band_features = np.zeros(self.num_bands, dtype=np.float32)

            return band_features

        except ValueError as e:
            # Expected errors: insufficient data, invalid values
            logger.debug(f"Error in PSD band extraction: {e}")
            return None
        except Exception as e:
            # Unexpected errors - log for investigation
            logger.error(f"Unexpected error in PSD band extraction: {e}", exc_info=True)
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

    def _extract_sfb21_features(self, S: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """SFB21 features for V0 vs LOSSLESS discrimination (length 6).

        Args:
            S: STFT magnitude matrix (n_freq x n_frames) from analyze_file
            freqs: Frequency array corresponding to S rows

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
        if S.size == 0:
            return np.zeros(6, dtype=np.float32)

        # Band masks
        below_sfb21 = (freqs >= 14000) & (freqs < 16000)  # Just below sfb21
        sfb21_band = (freqs >= 16000) & (freqs < 19500)  # sfb21 range
        ultra_band = (freqs >= 19500) & (freqs < 22000)  # Above V0 cutoff
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

        return np.array(
            [
                sfb21_ultra_ratio,
                sfb21_continuity,
                sfb21_flatness,
                sfb21_flat_std,
                sfb21_flat_iqr,
                flat_19_20k,
            ],
            dtype=np.float32,
        )

    def _extract_rolloff_features(self, S_power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Rolloff curve shape features between 18-21kHz (length 4).

        Args:
            S_power: STFT power matrix (|S|², n_freq x n_frames) from analyze_file
            freqs: Frequency array corresponding to S_power rows

        Features:
        1. rolloff_slope: Linear slope in dB/kHz - V0 steeper (more negative)
        2. rolloff_total_drop: Total dB drop across 18-21kHz - V0 drops more
        3. rolloff_ratio_early: 19-20kHz vs 18-19kHz energy - V0 higher
        4. rolloff_ratio_late: 20-21kHz vs 19-20kHz energy - LOSSLESS higher
        """
        if S_power.size == 0:
            return np.zeros(4, dtype=np.float32)

        # Mean power spectrum across time
        power_spectrum = np.mean(S_power, axis=1)

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

        return np.array(
            [
                rolloff_slope,
                rolloff_total_drop,
                rolloff_ratio_early,
                rolloff_ratio_late,
            ],
            dtype=np.float32,
        )

    def _extract_discriminative_features(
        self, psd_bands: np.ndarray, sfb21_features: np.ndarray
    ) -> np.ndarray:
        """Extract 6 discriminative features for 128/V2 class separation.

        Computed from already-extracted PSD bands and SFB21 features — no
        additional FFT/Welch calls needed.

        128 kbps discrimination (4 features):
        - f128_psd_ratio_low_high: mean(psd[0:30]) / mean(psd[70:100])
        - f128_psd_ratio_mid_ultra: mean(psd[40:60]) / mean(psd[100:130])
        - f128_energy_above_17k: sum(psd[25:100])
        - f128_energy_above_19k: sum(psd[75:100])

        V2/V0 discrimination (2 features):
        - v2_energy_ratio_19k: energy above 19k / energy below 19k
        - v2_sfb21_peak_ratio: max / mean in SFB21 region
        """
        eps = 1e-10
        MAX_RATIO = 20.0  # Beyond this, "very different" is enough info

        # Clamp to 0-1 range (defensive: flat-spectrum edge case)
        psd_bands = np.clip(psd_bands, 0.0, 1.0)

        # --- 128 kbps features ---
        mean_low = np.mean(psd_bands[0:30]) + eps
        mean_high = np.mean(psd_bands[70:100]) + eps
        f128_psd_ratio_low_high = np.clip(mean_low / mean_high, 0, MAX_RATIO)

        mean_mid = np.mean(psd_bands[40:60]) + eps
        mean_ultra = np.mean(psd_bands[100:130]) + eps
        f128_psd_ratio_mid_ultra = np.clip(mean_mid / mean_ultra, 0, MAX_RATIO)

        f128_energy_above_17k = float(np.sum(psd_bands[25:100]))
        f128_energy_above_19k = float(np.sum(psd_bands[75:100]))

        # --- V2/V0 features ---
        # v2_energy_ratio_19k: energy above 19k / energy below 19k (within analysis band)
        energy_below_19k = np.sum(psd_bands[:75]) + eps
        energy_above_19k = np.sum(psd_bands[75:]) + eps
        v2_energy_ratio_19k = np.clip(energy_above_19k / energy_below_19k, 0, MAX_RATIO)

        # v2_sfb21_peak_ratio: max / mean in SFB21 region
        sfb21_mean = np.mean(sfb21_features) + eps
        sfb21_max = np.max(sfb21_features) + eps
        v2_sfb21_peak_ratio = np.clip(sfb21_max / sfb21_mean, 0, MAX_RATIO)

        return np.array(
            [
                f128_psd_ratio_low_high,
                f128_psd_ratio_mid_ultra,
                f128_energy_above_17k,
                f128_energy_above_19k,
                v2_energy_ratio_19k,
                v2_sfb21_peak_ratio,
            ],
            dtype=np.float32,
        )

    def _extract_temporal_features(
        self, S_mag: np.ndarray, freqs: np.ndarray, sr: int
    ) -> np.ndarray:
        """Extract 4 temporal artifact variance features using 0.5s segments.

        Measures how artifact levels vary over time — CBR has more uniform
        artifacts while VBR adapts quality per segment.
        """
        hop_length = self.fft_size // 4  # 2048
        frames_per_seg = max(1, int(sr / hop_length * 0.5))
        n_frames = S_mag.shape[1]
        n_segments = n_frames // frames_per_seg

        if n_segments < 3:
            return np.zeros(4, dtype=np.float32)

        # Frequency masks
        artifact_mask = (freqs >= 10000) & (freqs <= 16000)
        content_mask = (freqs >= 2000) & (freqs <= 16000)

        if np.sum(artifact_mask) < 2 or np.sum(content_mask) < 2:
            return np.zeros(4, dtype=np.float32)

        eps = 1e-10
        segment_flatness = np.zeros(n_segments)
        segment_complexity = np.zeros(n_segments)

        for i in range(n_segments):
            start = i * frames_per_seg
            end = start + frames_per_seg
            seg = S_mag[:, start:end]

            # Artifact proxy: spectral flatness in 10-16 kHz
            artifact_band = seg[artifact_mask, :]
            gmean = np.exp(np.mean(np.log(artifact_band + eps)))
            amean = np.mean(artifact_band) + eps
            segment_flatness[i] = gmean / amean

            # Complexity proxy: spectral flux in 2-16 kHz
            content_band = seg[content_mask, :]
            if content_band.shape[1] > 1:
                flux = np.mean(np.abs(np.diff(content_band, axis=1)))
            else:
                flux = 0.0
            segment_complexity[i] = flux

        # 1. Artifact variance
        artifact_variance = float(np.var(segment_flatness))

        # 2. Artifact IQR
        q75, q25 = np.percentile(segment_flatness, [75, 25])
        artifact_iqr = float(q75 - q25)

        # 3. Artifact range
        artifact_range = float(np.ptp(segment_flatness))

        # 4. Complexity-artifact correlation
        if np.std(segment_complexity) < eps or np.std(segment_flatness) < eps:
            complexity_corr = 0.0
        else:
            corr = np.corrcoef(segment_complexity, segment_flatness)[0, 1]
            complexity_corr = 0.0 if np.isnan(corr) else float(corr)

        return np.array(
            [artifact_variance, artifact_iqr, artifact_range, complexity_corr],
            dtype=np.float32,
        )

    @staticmethod
    def _get_band_energy_over_time(
        S_power: np.ndarray, stft_freqs: np.ndarray, lo: float, hi: float
    ) -> np.ndarray:
        """Get time-varying energy in a frequency band.

        Args:
            S_power: Power spectrogram (freq_bins x time_frames)
            stft_freqs: Frequency values for each bin
            lo: Lower frequency bound (Hz)
            hi: Upper frequency bound (Hz)

        Returns:
            1D array of energy per time frame
        """
        mask = (stft_freqs >= lo) & (stft_freqs < hi)
        if not np.any(mask):
            return np.zeros(S_power.shape[1], dtype=np.float32)
        return np.sum(S_power[mask, :], axis=0)

    def _extract_crossband_features(
        self, S_power: np.ndarray, stft_freqs: np.ndarray
    ) -> np.ndarray:
        """Extract cross-band correlation features.

        Measures how well upper frequency bands track musical content
        relative to a 1-4 kHz reference band. Higher quality encodings
        preserve cross-band correlation; lossy encoding decorrelates
        upper bands.

        Returns:
            6-element array: corr_veryhigh, corr_ultra, mod_veryhigh,
            mod_ultra, transient_veryhigh, corr_gradient
        """
        # All band boundaries: ref + 5 target bands (computed in one pass)
        band_defs = [
            (1000, 4000),    # ref
            (4000, 8000),    # mid (gradient only)
            (8000, 12000),   # upper (gradient only)
            (12000, 16000),  # high (gradient only)
            (16000, 19000),  # veryhigh
            (19000, 22000),  # ultra
        ]

        # Build index masks for all bands at once
        band_energies = []
        for lo, hi in band_defs:
            mask = (stft_freqs >= lo) & (stft_freqs < hi)
            if np.any(mask):
                band_energies.append(np.sum(S_power[mask, :], axis=0))
            else:
                band_energies.append(np.zeros(S_power.shape[1], dtype=np.float32))

        ref_energy = band_energies[0]
        ref_std = np.std(ref_energy)

        # Compute correlations for all 5 target bands (for gradient)
        all_corrs = []
        for i in range(1, 6):
            be = band_energies[i]
            if ref_std > 1e-10 and np.std(be) > 1e-10:
                c = np.corrcoef(ref_energy, be)[0, 1]
                all_corrs.append(0.0 if np.isnan(c) else c)
            else:
                all_corrs.append(0.0)

        corr_veryhigh = all_corrs[3]  # index 4 in band_defs = veryhigh
        corr_ultra = all_corrs[4]     # index 5 in band_defs = ultra

        # Modulation depth for veryhigh and ultra
        vh_energy = band_energies[4]
        ul_energy = band_energies[5]
        mod_veryhigh = np.std(vh_energy) / (np.mean(vh_energy) + 1e-10)
        mod_ultra = np.std(ul_energy) / (np.mean(ul_energy) + 1e-10)

        # Transient sharpness for veryhigh
        gradient = np.abs(np.diff(vh_energy))
        grad_mean = np.mean(gradient)
        transient_veryhigh = (
            np.percentile(gradient, 95) / (grad_mean + 1e-10)
            if grad_mean > 1e-10
            else 0.0
        )

        # Correlation gradient (slope of correlation vs frequency band index)
        corr_gradient = np.polyfit(range(5), all_corrs, 1)[0]

        return np.array(
            [corr_veryhigh, corr_ultra, mod_veryhigh, mod_ultra,
             transient_veryhigh, corr_gradient],
            dtype=np.float32,
        )

    def _extract_cutoff_cleanliness_features(
        self,
        S_power: np.ndarray,
        S_mag: np.ndarray,
        stft_freqs: np.ndarray,
        cutoff_features: np.ndarray,
    ) -> np.ndarray:
        """Extract 5 cutoff cleanliness features adaptive to detected cutoff.

        Measures artifact behavior around the detected cutoff frequency.
        Adapts to wherever the cutoff actually is (128->16kHz, 192->18kHz, etc.).
        For V0/FLAC with no detectable cutoff, uses 20kHz as reference point.

        Args:
            S_power: Power spectrogram (freq_bins x time_frames)
            S_mag: Magnitude spectrogram (freq_bins x time_frames)
            stft_freqs: Frequency values for each STFT bin
            cutoff_features: 6-element cutoff features (index 0 = primary_cutoff normalized)

        Returns:
            5-element array: bleed_ratio, floor, variance, edge_gradient,
            correlation
        """
        if S_power.ndim < 2 or S_power.shape[1] < 2:
            return np.zeros(5, dtype=np.float32)

        # Convert normalized primary_cutoff (0-1 in 16-22kHz) back to Hz
        primary_cutoff_norm = float(cutoff_features[0])
        cutoff_hz = primary_cutoff_norm * (self.max_freq - self.min_freq) + self.min_freq

        # If no cutoff detected, use 20kHz as reference
        if cutoff_hz > 20000 or primary_cutoff_norm < 0.01:
            cutoff_hz = 20000.0

        # Define adaptive bands around cutoff
        below_lo, below_hi = cutoff_hz - 1500, cutoff_hz - 500
        above_lo, above_hi = cutoff_hz + 500, cutoff_hz + 1500

        # Per-frame energy in below and above bands
        below_energy = self._get_band_energy_over_time(S_power, stft_freqs, below_lo, below_hi)
        above_energy = self._get_band_energy_over_time(S_power, stft_freqs, above_lo, above_hi)

        # Feature 1: Cutoff bleed ratio (above/below total energy)
        below_sum = float(np.sum(below_energy))
        above_sum = float(np.sum(above_energy))
        cutoff_bleed_ratio = float(np.clip(above_sum / (below_sum + 1e-10), 0.0, 20.0))

        # Feature 2: Above-cutoff noise floor (25th percentile, normalized)
        above_mask = (stft_freqs >= above_lo) & (stft_freqs < above_hi)
        if np.any(above_mask):
            above_mags = S_mag[above_mask, :].flatten()
            floor_val = float(np.percentile(above_mags, 25))
            overall_median = float(np.median(S_mag) + 1e-10)
            above_cutoff_floor = float(np.clip(floor_val / overall_median, 0.0, 10.0))
        else:
            above_cutoff_floor = 0.0

        # Feature 3: Above-cutoff variance (scale-invariant)
        mean_above = float(np.mean(above_energy))
        if mean_above > 1e-10:
            above_cutoff_variance = float(np.clip(np.var(above_energy / mean_above), 0.0, 50.0))
        else:
            above_cutoff_variance = 0.0

        # Feature 4: Cutoff edge gradient (max step in transition zone)
        transition_lo = cutoff_hz - 1000
        transition_hi = cutoff_hz + 1000
        n_bands = 20
        band_edges = np.linspace(transition_lo, transition_hi, n_bands + 1)
        transition_energies = np.zeros(n_bands, dtype=np.float32)
        for i in range(n_bands):
            mask = (stft_freqs >= band_edges[i]) & (stft_freqs < band_edges[i + 1])
            if np.any(mask):
                transition_energies[i] = float(np.mean(S_power[mask, :]))
        te_max = float(transition_energies.max())
        if te_max > 1e-10:
            transition_energies /= te_max
        diffs = np.abs(np.diff(transition_energies))
        cutoff_edge_gradient = float(np.max(diffs)) if len(diffs) > 0 else 0.0

        # Feature 5: Above/below correlation
        if np.std(below_energy) > 1e-10 and np.std(above_energy) > 1e-10:
            corr = np.corrcoef(below_energy, above_energy)[0, 1]
            cutoff_correlation = float(corr) if not np.isnan(corr) else 0.0
        else:
            cutoff_correlation = 0.0

        return np.array(
            [
                cutoff_bleed_ratio,
                above_cutoff_floor,
                above_cutoff_variance,
                cutoff_edge_gradient,
                cutoff_correlation,
            ],
            dtype=np.float32,
        )

    def _extract_mdct_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract 6 MDCT forensic features based on zero-coefficient analysis.

        Computes MDCT coefficients using the MP3 long block window size (1152)
        and extracts statistics about near-zero coefficients that reveal
        encoder quantization patterns. These features survive the
        decode→PCM→re-MDCT round-trip because they measure offset-independent
        statistical properties of the coefficient magnitudes.

        Features (6 total):
        - mdct_zero_ratio_mean/var/iqr: near-zero coefficient statistics
        - mdct_sfb21_zero_mean/var: high-frequency zero patterns
        - mdct_sfb21_lower_ratio: SFB21 vs lower-band zero ratio

        Args:
            y: Audio time series (mono)
            sr: Sample rate in Hz

        Returns:
            6-element float32 array
        """
        N_WINDOW = 1152  # MP3 long block window size
        HALF_N = N_WINDOW // 2  # 576 = hop and MDCT output size

        # Use first 10 seconds for efficiency
        max_samples = min(len(y), sr * 10)
        audio = np.ascontiguousarray(y[:max_samples], dtype=np.float64)

        if len(audio) < N_WINDOW * 4:
            return np.zeros(6, dtype=np.float32)

        # MP3 sine window
        window = np.sin(np.pi * (np.arange(N_WINDOW) + 0.5) / N_WINDOW)

        # ── MDCT coefficients at offset 0 ──────────────────────────
        seg = np.ascontiguousarray(audio)
        if len(seg) < N_WINDOW:
            return np.zeros(6, dtype=np.float32)

        frames_T = librosa.util.frame(
            seg, frame_length=N_WINDOW, hop_length=HALF_N
        )
        n_frames = min(frames_T.shape[1], 400)  # Cap for memory

        if n_frames < 4:
            return np.zeros(6, dtype=np.float32)

        frames = frames_T[:, :n_frames].T  # (n_frames, N_WINDOW)
        windowed = frames * window

        # MDCT basis: X[k] = Σ x[n]·cos(π/N·(n+0.5+N/2)·(k+0.5))
        n_idx = np.arange(N_WINDOW)[:, np.newaxis]  # (1152, 1)
        k_idx = np.arange(HALF_N)[np.newaxis, :]  # (1, 576)
        basis = np.cos(
            np.pi / HALF_N * (n_idx + 0.5 + HALF_N / 2) * (k_idx + 0.5)
        )

        coeffs = windowed @ basis  # (n_frames, 576)

        # Near-zero threshold: 1% of median absolute coefficient
        median_abs = np.median(np.abs(coeffs))
        threshold = max(median_abs * 0.01, 1e-10)

        # ── Features 1-3: Zero coefficient ratio ───────────────────
        zero_mask = np.abs(coeffs) < threshold
        zero_ratio_per_frame = np.mean(zero_mask, axis=1)
        zero_ratio_mean = float(np.mean(zero_ratio_per_frame))
        zero_ratio_var = float(np.var(zero_ratio_per_frame))
        q75, q25 = np.percentile(zero_ratio_per_frame, [75, 25])
        zero_ratio_iqr = float(q75 - q25)

        # ── Features 4-6: SFB21 zero patterns ──────────────────────
        sfb21_size = 50  # Last 50 coefficients (~20-22 kHz)
        sfb21_zeros = zero_mask[:, -sfb21_size:]
        lower_zeros = zero_mask[:, :-sfb21_size]
        sfb21_zero_per_frame = np.mean(sfb21_zeros, axis=1)
        lower_zero_per_frame = np.mean(lower_zeros, axis=1)

        sfb21_zero_mean = float(np.mean(sfb21_zero_per_frame))
        sfb21_zero_var = float(np.var(sfb21_zero_per_frame))
        sfb21_lower_ratio = float(
            sfb21_zero_mean / (float(np.mean(lower_zero_per_frame)) + 1e-10)
        )

        # ── Clamp outlier-prone feature ─────────────────────────────
        sfb21_lower_ratio = float(np.clip(sfb21_lower_ratio, 0, 50))

        return np.array(
            [
                zero_ratio_mean,
                zero_ratio_var,
                zero_ratio_iqr,
                sfb21_zero_mean,
                sfb21_zero_var,
                sfb21_lower_ratio,
            ],
            dtype=np.float32,
        )

    def _split_feature_vector(
        self, vector: np.ndarray, metadata: dict
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ]:
        """Split cached feature vector into component arrays."""
        psd_len = metadata.get("n_bands", self.num_bands)
        cutoff_len = metadata.get("cutoff_len", 6)
        sfb21_len = metadata.get("sfb21_len", 6)
        rolloff_len = metadata.get("rolloff_len", 4)
        discriminative_len = metadata.get("discriminative_len", 6)
        temporal_len = metadata.get("temporal_len", 0)
        crossband_len = metadata.get("crossband_len", 0)
        cutoff_cleanliness_len = metadata.get("cutoff_cleanliness_len", 0)
        mdct_len = metadata.get("mdct_len", 0)

        psd_end = psd_len
        cutoff_end = psd_end + cutoff_len
        sfb21_end = cutoff_end + sfb21_len
        rolloff_end = sfb21_end + rolloff_len
        discriminative_end = rolloff_end + discriminative_len
        temporal_end = discriminative_end + temporal_len
        crossband_end = temporal_end + crossband_len
        cutoff_cleanliness_end = crossband_end + cutoff_cleanliness_len
        mdct_end = cutoff_cleanliness_end + mdct_len

        psd_bands = vector[:psd_end]
        cutoff_feats = vector[psd_end:cutoff_end] if cutoff_len else np.zeros(6, dtype=np.float32)
        sfb21_feats = vector[cutoff_end:sfb21_end] if sfb21_len else np.zeros(6, dtype=np.float32)
        rolloff_feats = (
            vector[sfb21_end:rolloff_end] if rolloff_len else np.zeros(4, dtype=np.float32)
        )
        discriminative_feats = (
            vector[rolloff_end:discriminative_end]
            if discriminative_len
            else np.zeros(6, dtype=np.float32)
        )
        temporal_feats = (
            vector[discriminative_end:temporal_end]
            if temporal_len
            else np.zeros(4, dtype=np.float32)
        )
        crossband_feats = (
            vector[temporal_end:crossband_end]
            if crossband_len
            else np.zeros(6, dtype=np.float32)
        )
        cutoff_cleanliness_feats = (
            vector[crossband_end:cutoff_cleanliness_end]
            if cutoff_cleanliness_len
            else np.zeros(5, dtype=np.float32)
        )
        mdct_feats = (
            vector[cutoff_cleanliness_end:mdct_end]
            if mdct_len
            else np.zeros(6, dtype=np.float32)
        )
        return (
            np.asarray(psd_bands, dtype=np.float32),
            np.asarray(cutoff_feats, dtype=np.float32),
            np.asarray(sfb21_feats, dtype=np.float32),
            np.asarray(rolloff_feats, dtype=np.float32),
            np.asarray(discriminative_feats, dtype=np.float32),
            np.asarray(temporal_feats, dtype=np.float32),
            np.asarray(crossband_feats, dtype=np.float32),
            np.asarray(cutoff_cleanliness_feats, dtype=np.float32),
            np.asarray(mdct_feats, dtype=np.float32),
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
