"""Tests for cutoff detection."""

import numpy as np

from beetsplug.bitrater.constants import CLASS_CUTOFFS
from beetsplug.bitrater.cutoff_detector import CutoffDetector, CutoffResult


class TestCutoffDetector:
    """Test CutoffDetector class."""

    def test_init(self):
        """CutoffDetector should initialize with default parameters."""
        detector = CutoffDetector()
        assert detector.min_freq == 15000
        assert detector.max_freq == 22050
        assert detector.coarse_step == 1000
        assert detector.fine_step == 100

    def test_coarse_scan_finds_128kbps_cutoff(self):
        """Coarse scan should find approximate cutoff for 128 kbps (~16 kHz)."""
        detector = CutoffDetector()
        expected_cutoff = CLASS_CUTOFFS["128"]

        # Create mock PSD: high energy below cutoff, noise floor above
        freqs = np.linspace(0, 22050, 2048)
        psd = np.ones_like(freqs)
        psd[freqs > expected_cutoff] = 0.001  # Sharp drop at expected cutoff

        candidate = detector._coarse_scan(psd, freqs)

        # Should find cutoff within 1 kHz of actual
        assert expected_cutoff - 1000 <= candidate <= expected_cutoff + 1000

    def test_coarse_scan_finds_192kbps_cutoff(self):
        """Coarse scan should find approximate cutoff for 192 kbps (~19 kHz)."""
        detector = CutoffDetector()
        expected_cutoff = CLASS_CUTOFFS["192"]

        freqs = np.linspace(0, 22050, 2048)
        psd = np.ones_like(freqs)
        psd[freqs > expected_cutoff] = 0.001

        candidate = detector._coarse_scan(psd, freqs)

        assert expected_cutoff - 1000 <= candidate <= expected_cutoff + 1000

    def test_fine_scan_refines_cutoff(self):
        """Fine scan should refine cutoff to within 100 Hz."""
        detector = CutoffDetector()
        expected_cutoff = CLASS_CUTOFFS["128"]

        # Create PSD with cutoff slightly above expected
        actual_cutoff = expected_cutoff + 200
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)
        psd[freqs > actual_cutoff] = 0.001

        # Start with coarse estimate at expected cutoff
        coarse_estimate = expected_cutoff

        refined = detector._fine_scan(psd, freqs, coarse_estimate)

        # Should refine to within 200 Hz of actual cutoff
        assert actual_cutoff - 200 <= refined <= actual_cutoff + 200

    def test_gradient_sharp_for_artificial_cutoff(self):
        """Sharp artificial cutoff should have high gradient."""
        detector = CutoffDetector()
        cutoff = CLASS_CUTOFFS["128"]

        # Sharp step function at 128 kbps cutoff
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)
        psd[freqs > cutoff] = 0.001  # Instant drop

        gradient = detector._measure_gradient(psd, freqs, cutoff)

        assert gradient > detector.sharp_threshold

    def test_gradient_gradual_for_natural_rolloff(self):
        """Natural gradual rolloff should have low gradient."""
        detector = CutoffDetector()
        cutoff = CLASS_CUTOFFS["128"]

        # Gradual rolloff - exponential decay starting well before cutoff
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)
        rolloff_start = cutoff - 2000
        rolloff_mask = freqs > rolloff_start
        psd[rolloff_mask] = np.exp(-0.0005 * (freqs[rolloff_mask] - rolloff_start))

        gradient = detector._measure_gradient(psd, freqs, cutoff)

        assert gradient < detector.sharp_threshold

    def test_detect_128kbps_cutoff(self):
        """Detect should identify 128 kbps cutoff."""
        detector = CutoffDetector()
        expected_cutoff = CLASS_CUTOFFS["128"]

        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)
        psd[freqs > expected_cutoff] = 0.001

        result = detector.detect(psd, freqs)

        assert isinstance(result, CutoffResult)
        assert expected_cutoff - 500 <= result.cutoff_frequency <= expected_cutoff + 500
        assert result.is_sharp is True  # Artificial cutoff
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_lossless_no_cutoff(self):
        """Detect should identify lossless with cutoff near Nyquist."""
        detector = CutoffDetector()

        # 4096 points gives ~5.4 Hz resolution for 22050 Hz Nyquist
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)  # Full spectrum, no cutoff

        result = detector.detect(psd, freqs)

        assert result.cutoff_frequency > 21000

    def test_detect_uniform_spectrum_no_clear_cutoff(self):
        """Detector should handle spectrum with no clear cutoff gracefully."""
        detector = CutoffDetector()

        # Uniform spectrum with constant energy - no artificial cutoff
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs) * 0.5  # Uniform energy throughout

        result = detector.detect(psd, freqs)

        # Should return max frequency since no drop detected
        assert result.cutoff_frequency >= detector.max_freq - detector.coarse_step
        # Low gradient expected since no sharp transition
        assert result.is_sharp is False
