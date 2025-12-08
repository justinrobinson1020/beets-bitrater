"""Tests for confidence calculation with penalties."""

import pytest

from beetsplug.bitrater.confidence import ConfidenceCalculator


class TestConfidenceCalculator:
    """Test confidence penalty calculations."""

    def test_no_penalty_when_cutoff_matches(self):
        """No penalty when detected cutoff matches expected."""
        calc = ConfidenceCalculator()

        # 320 kbps expects 20500 Hz, detected 20400 Hz (within tolerance)
        result = calc.calculate(
            classifier_confidence=0.9,
            detected_class="320",
            detected_cutoff=20400,
            gradient=0.8,  # Sharp
        )

        assert result.final_confidence >= 0.85  # Minimal penalty
        assert len(result.warnings) == 0

    def test_penalty_for_cutoff_mismatch(self):
        """Penalty applied when cutoff doesn't match class."""
        calc = ConfidenceCalculator()

        # Classifier says 320 (expects 20500 Hz), but cutoff is 19000 Hz (192 range)
        result = calc.calculate(
            classifier_confidence=0.9,
            detected_class="320",
            detected_cutoff=19000,
            gradient=0.8,
        )

        assert result.final_confidence < 0.7  # Significant penalty
        assert any("mismatch" in w.lower() for w in result.warnings)

    def test_penalty_for_soft_gradient(self):
        """Penalty applied for gradual rolloff (natural, not artificial)."""
        calc = ConfidenceCalculator()

        result = calc.calculate(
            classifier_confidence=0.9,
            detected_class="128",
            detected_cutoff=16000,
            gradient=0.2,  # Soft gradient
        )

        assert result.final_confidence < 0.85
        assert any("rolloff" in w.lower() or "natural" in w.lower() for w in result.warnings)

    def test_minimum_confidence_floor(self):
        """Confidence should never go below 0.1."""
        calc = ConfidenceCalculator()

        result = calc.calculate(
            classifier_confidence=0.3,  # Low to start
            detected_class="LOSSLESS",
            detected_cutoff=16000,  # Way off (should be >21500)
            gradient=0.2,  # Soft
        )

        assert result.final_confidence >= 0.1
