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
