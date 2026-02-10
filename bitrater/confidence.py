"""Confidence calculation with penalty system."""

from dataclasses import dataclass, field

from .constants import CLASS_CUTOFFS, CUTOFF_TOLERANCE


@dataclass
class ConfidenceResult:
    """Result of confidence calculation."""

    final_confidence: float
    mismatch_penalty: float
    gradient_penalty: float
    warnings: list[str] = field(default_factory=list)


class ConfidenceCalculator:
    """
    Calculate final confidence using primary + penalties approach.

    Starts with classifier confidence, subtracts penalties for:
    - Cutoff mismatch (detected cutoff doesn't match expected for class)
    - Soft gradient (gradual rolloff suggests natural, not encoded)
    """

    def __init__(
        self,
        mismatch_penalty_small: float = 0.10,
        mismatch_penalty_medium: float = 0.25,
        mismatch_penalty_large: float = 0.40,
        gradient_penalty_moderate: float = 0.10,
        gradient_penalty_soft: float = 0.20,
        sharp_threshold: float = 0.5,
        minimum_confidence: float = 0.1,
    ):
        """Initialize with penalty values."""
        self.mismatch_penalty_small = mismatch_penalty_small
        self.mismatch_penalty_medium = mismatch_penalty_medium
        self.mismatch_penalty_large = mismatch_penalty_large
        self.gradient_penalty_moderate = gradient_penalty_moderate
        self.gradient_penalty_soft = gradient_penalty_soft
        self.sharp_threshold = sharp_threshold
        self.minimum_confidence = minimum_confidence

    def calculate(
        self,
        classifier_confidence: float,
        detected_class: str,
        detected_cutoff: int,
        gradient: float,
    ) -> ConfidenceResult:
        """
        Calculate final confidence with penalties.

        Args:
            classifier_confidence: Raw confidence from SVM classifier
            detected_class: Predicted class from classifier
            detected_cutoff: Cutoff frequency from cutoff detector
            gradient: Gradient sharpness from cutoff detector

        Returns:
            ConfidenceResult with final confidence and warnings
        """
        warnings: list[str] = []

        # Calculate mismatch penalty
        mismatch_penalty = self._calculate_mismatch_penalty(
            detected_class, detected_cutoff, warnings
        )

        # Calculate gradient penalty
        gradient_penalty = self._calculate_gradient_penalty(gradient, warnings)

        # Apply penalties
        final = classifier_confidence - mismatch_penalty - gradient_penalty
        final = max(self.minimum_confidence, final)

        return ConfidenceResult(
            final_confidence=final,
            mismatch_penalty=mismatch_penalty,
            gradient_penalty=gradient_penalty,
            warnings=warnings,
        )

    def _calculate_mismatch_penalty(
        self,
        detected_class: str,
        detected_cutoff: int,
        warnings: list[str],
    ) -> float:
        """Calculate penalty for cutoff mismatch."""
        if detected_class not in CLASS_CUTOFFS:
            return 0.0

        expected_cutoff = CLASS_CUTOFFS[detected_class]
        difference = abs(detected_cutoff - expected_cutoff)

        if difference <= CUTOFF_TOLERANCE:
            return 0.0
        elif difference <= CUTOFF_TOLERANCE * 2:
            return self.mismatch_penalty_small
        elif difference <= CUTOFF_TOLERANCE * 4:
            warnings.append(
                f"Cutoff mismatch: expected ~{expected_cutoff} Hz, "
                f"found {detected_cutoff} Hz"
            )
            return self.mismatch_penalty_medium
        else:
            warnings.append(
                f"Significant cutoff mismatch: expected ~{expected_cutoff} Hz, "
                f"found {detected_cutoff} Hz"
            )
            return self.mismatch_penalty_large

    def _calculate_gradient_penalty(
        self,
        gradient: float,
        warnings: list[str],
    ) -> float:
        """Calculate penalty for soft gradient (natural rolloff)."""
        if gradient >= self.sharp_threshold:
            return 0.0
        elif gradient >= self.sharp_threshold * 0.5:
            return self.gradient_penalty_moderate
        else:
            warnings.append(
                "Gradual frequency rolloff detected - may be natural, not encoded"
            )
            return self.gradient_penalty_soft
