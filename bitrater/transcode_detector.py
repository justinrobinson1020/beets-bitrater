"""Transcode detection based on quality ranking."""

from dataclasses import dataclass

from .constants import QUALITY_RANK


@dataclass
class TranscodeResult:
    """Result of transcode detection."""

    is_transcode: bool
    quality_gap: int  # 0-6, higher = more severe
    transcoded_from: str | None  # Original quality if transcode


class TranscodeDetector:
    """
    Detect transcodes by comparing stated vs detected quality.

    Transcode = stated quality rank > detected quality rank
    """

    def detect(
        self,
        stated_class: str,
        detected_class: str,
    ) -> TranscodeResult:
        """
        Determine if file is a transcode.

        Args:
            stated_class: What the file claims to be (from container/metadata)
            detected_class: What the classifier detected

        Returns:
            TranscodeResult with detection info
        """
        stated_rank = QUALITY_RANK.get(stated_class, 0)
        detected_rank = QUALITY_RANK.get(detected_class, 0)

        is_transcode = stated_rank > detected_rank
        quality_gap = stated_rank - detected_rank if is_transcode else 0
        transcoded_from = detected_class if is_transcode else None

        return TranscodeResult(
            is_transcode=is_transcode,
            quality_gap=quality_gap,
            transcoded_from=transcoded_from,
        )
