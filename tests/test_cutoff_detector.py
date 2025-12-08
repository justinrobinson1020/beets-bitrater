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
