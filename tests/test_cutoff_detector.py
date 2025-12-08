"""Tests for cutoff detection."""

import numpy as np
import pytest

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

    def test_coarse_scan_finds_16khz_cutoff(self):
        """Coarse scan should find approximate cutoff at 16 kHz."""
        detector = CutoffDetector()

        # Create mock PSD: high energy below 16 kHz, noise floor above
        freqs = np.linspace(0, 22050, 2048)
        psd = np.ones_like(freqs)
        psd[freqs > 16000] = 0.001  # Sharp drop at 16 kHz

        candidate = detector._coarse_scan(psd, freqs)

        # Should find cutoff within 1 kHz of actual
        assert 15000 <= candidate <= 17000

    def test_coarse_scan_finds_19khz_cutoff(self):
        """Coarse scan should find approximate cutoff at 19 kHz."""
        detector = CutoffDetector()

        freqs = np.linspace(0, 22050, 2048)
        psd = np.ones_like(freqs)
        psd[freqs > 19000] = 0.001

        candidate = detector._coarse_scan(psd, freqs)

        assert 18000 <= candidate <= 20000

    def test_fine_scan_refines_cutoff(self):
        """Fine scan should refine cutoff to within 100 Hz."""
        detector = CutoffDetector()

        # Create PSD with cutoff at exactly 16200 Hz
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)
        psd[freqs > 16200] = 0.001

        # Start with coarse estimate
        coarse_estimate = 16000

        refined = detector._fine_scan(psd, freqs, coarse_estimate)

        # Should refine to within 200 Hz of actual cutoff
        assert 16000 <= refined <= 16400

    def test_gradient_sharp_for_artificial_cutoff(self):
        """Sharp artificial cutoff should have high gradient."""
        detector = CutoffDetector()

        # Sharp step function at 16 kHz
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)
        psd[freqs > 16000] = 0.001  # Instant drop

        gradient = detector._measure_gradient(psd, freqs, 16000)

        assert gradient > detector.sharp_threshold

    def test_gradient_gradual_for_natural_rolloff(self):
        """Natural gradual rolloff should have low gradient."""
        detector = CutoffDetector()

        # Gradual rolloff - exponential decay starting at 14 kHz
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)
        rolloff_mask = freqs > 14000
        psd[rolloff_mask] = np.exp(-0.0005 * (freqs[rolloff_mask] - 14000))

        gradient = detector._measure_gradient(psd, freqs, 16000)

        assert gradient < detector.sharp_threshold
