"""Tests for transcode detection logic."""


from beetsplug.bitrater.transcode_detector import TranscodeDetector


class TestTranscodeDetector:
    """Test transcode detection based on quality ranking."""

    def test_flac_from_128_is_transcode(self):
        """FLAC with 128 kbps content should be detected as transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="LOSSLESS",
            detected_class="128",
        )

        assert result.is_transcode is True
        assert result.quality_gap == 6  # LOSSLESS(6) - 128(0)
        assert result.transcoded_from == "128"

    def test_mp3_320_from_192_is_transcode(self):
        """320 kbps MP3 with 192 kbps content is transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="192",
        )

        assert result.is_transcode is True
        assert result.quality_gap == 3  # 320(5) - 192(2)
        assert result.transcoded_from == "192"

    def test_genuine_320_is_not_transcode(self):
        """Genuine 320 kbps file is not a transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="320",
        )

        assert result.is_transcode is False
        assert result.quality_gap == 0
        assert result.transcoded_from is None

    def test_192_detected_as_320_is_not_transcode(self):
        """File claiming lower quality than detected is not transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="192",
            detected_class="320",
        )

        assert result.is_transcode is False
        assert result.quality_gap == 0  # No gap when detected > stated
