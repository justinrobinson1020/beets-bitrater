"""End-to-end integration tests for transcode detection."""

import numpy as np
import pytest

from beetsplug.bitrater.cutoff_detector import CutoffDetector
from beetsplug.bitrater.confidence import ConfidenceCalculator
from beetsplug.bitrater.transcode_detector import TranscodeDetector
from beetsplug.bitrater.constants import QUALITY_RANK, CLASS_CUTOFFS


class TestFullPipeline:
    """Test complete transcode detection pipeline."""

    def test_128_to_flac_detection_pipeline(self):
        """Full pipeline should detect 128 kbps source in FLAC container."""
        # Simulate 128 kbps spectral signature
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)
        psd[freqs > 16000] = 0.001  # 128 kbps cutoff

        # Step 1: Cutoff detection
        cutoff_detector = CutoffDetector()
        cutoff_result = cutoff_detector.detect(psd, freqs)

        assert 15500 <= cutoff_result.cutoff_frequency <= 16500
        assert cutoff_result.is_sharp is True

        # Step 2: Confidence calculation (simulating classifier said "128")
        conf_calc = ConfidenceCalculator()
        conf_result = conf_calc.calculate(
            classifier_confidence=0.9,
            detected_class="128",
            detected_cutoff=cutoff_result.cutoff_frequency,
            gradient=cutoff_result.gradient,
        )

        # Cutoff matches, so no major penalty
        assert conf_result.final_confidence >= 0.8

        # Step 3: Transcode detection (FLAC stated, 128 detected)
        transcode_detector = TranscodeDetector()
        transcode_result = transcode_detector.detect(
            stated_class="LOSSLESS",
            detected_class="128",
        )

        assert transcode_result.is_transcode is True
        assert transcode_result.quality_gap == 6
        assert transcode_result.transcoded_from == "128"

    def test_genuine_lossless_pipeline(self):
        """Full pipeline should correctly identify genuine lossless."""
        # Simulate lossless spectral signature (full spectrum)
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)  # Full spectrum, no artificial cutoff

        cutoff_detector = CutoffDetector()
        cutoff_result = cutoff_detector.detect(psd, freqs)

        # Should find cutoff at or above 21 kHz
        assert cutoff_result.cutoff_frequency > 20000

        # Transcode detection (FLAC stated, LOSSLESS detected)
        transcode_detector = TranscodeDetector()
        transcode_result = transcode_detector.detect(
            stated_class="LOSSLESS",
            detected_class="LOSSLESS",
        )

        assert transcode_result.is_transcode is False
        assert transcode_result.quality_gap == 0
