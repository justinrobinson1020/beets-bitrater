"""Tests for confidence calculation with penalties."""


from beetsplug.bitrater.confidence import ConfidenceCalculator
from beetsplug.bitrater.constants import CLASS_CUTOFFS, CUTOFF_TOLERANCE


class TestConfidenceCalculator:
    """Test confidence penalty calculations."""

    def test_no_penalty_when_cutoff_matches(self):
        """No penalty when detected cutoff matches expected within tolerance."""
        calc = ConfidenceCalculator()
        expected_cutoff = CLASS_CUTOFFS["320"]

        # Detected cutoff within CUTOFF_TOLERANCE of expected
        detected_cutoff = expected_cutoff - (CUTOFF_TOLERANCE // 2)
        result = calc.calculate(
            classifier_confidence=0.9,
            detected_class="320",
            detected_cutoff=detected_cutoff,
            gradient=0.8,  # Sharp
        )

        assert result.final_confidence >= 0.85  # Minimal penalty
        assert len(result.warnings) == 0

    def test_penalty_for_cutoff_mismatch(self):
        """Penalty applied when cutoff doesn't match class."""
        calc = ConfidenceCalculator()

        # Classifier says 320, but cutoff matches 192 kbps range
        result = calc.calculate(
            classifier_confidence=0.9,
            detected_class="320",
            detected_cutoff=CLASS_CUTOFFS["192"],  # Mismatch with claimed "320"
            gradient=0.8,
        )

        # Significant mismatch (>4x CUTOFF_TOLERANCE) should trigger warning
        assert result.final_confidence < 0.7  # Significant penalty
        assert any("mismatch" in w.lower() for w in result.warnings)

    def test_penalty_for_soft_gradient(self):
        """Penalty applied for gradual rolloff (natural, not artificial)."""
        calc = ConfidenceCalculator()
        expected_cutoff = CLASS_CUTOFFS["128"]

        result = calc.calculate(
            classifier_confidence=0.9,
            detected_class="128",
            detected_cutoff=expected_cutoff,
            gradient=0.2,  # Soft gradient (below sharp_threshold of 0.5)
        )

        assert result.final_confidence < 0.85
        assert any("rolloff" in w.lower() or "natural" in w.lower() for w in result.warnings)

    def test_minimum_confidence_floor(self):
        """Confidence should never go below minimum (0.1)."""
        calc = ConfidenceCalculator()

        # Major mismatch: classified as lossless but cutoff is at 128 kbps level
        result = calc.calculate(
            classifier_confidence=0.3,  # Low to start
            detected_class="LOSSLESS",
            detected_cutoff=CLASS_CUTOFFS["128"],  # Way off from LOSSLESS expectation
            gradient=0.2,  # Soft
        )

        assert result.final_confidence >= calc.minimum_confidence
