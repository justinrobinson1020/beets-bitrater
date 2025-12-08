"""Tests for transcode detection logic."""

from bitrater.constants import QUALITY_RANK
from bitrater.transcode_detector import TranscodeDetector


class TestTranscodeDetector:
    """Test transcode detection based on quality ranking."""

    def test_flac_from_128_is_transcode(self) -> None:
        """FLAC with 128 kbps content should be detected as transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="LOSSLESS",
            detected_class="128",
        )

        expected_gap = QUALITY_RANK["LOSSLESS"] - QUALITY_RANK["128"]
        assert result.is_transcode is True
        assert result.quality_gap == expected_gap
        assert result.transcoded_from == "128"

    def test_mp3_320_from_192_is_transcode(self) -> None:
        """320 kbps MP3 with 192 kbps content is transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="192",
        )

        expected_gap = QUALITY_RANK["320"] - QUALITY_RANK["192"]
        assert result.is_transcode is True
        assert result.quality_gap == expected_gap
        assert result.transcoded_from == "192"

    def test_genuine_320_is_not_transcode(self) -> None:
        """Genuine 320 kbps file is not a transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="320",
        )

        assert result.is_transcode is False
        assert result.quality_gap == 0
        assert result.transcoded_from is None

    def test_192_detected_as_320_is_not_transcode(self) -> None:
        """File claiming lower quality than detected is not transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="192",
            detected_class="320",
        )

        assert result.is_transcode is False
        assert result.quality_gap == 0  # No gap when detected > stated
