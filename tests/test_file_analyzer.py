"""Tests for file metadata extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from mutagen.mp3 import BitrateMode

from bitrater.file_analyzer import FileAnalyzer


class TestFileAnalyzerDispatch:
    """Tests for analyze() routing by file extension."""

    def test_mp3_routes_to_analyze_mp3(self, tmp_path: Path) -> None:
        """analyze() should route .mp3 files to _analyze_mp3."""
        fa = FileAnalyzer()
        f = tmp_path / "test.mp3"
        f.write_bytes(b"fake")

        with patch.object(fa, "_analyze_mp3", return_value=Mock()) as mock:
            fa.analyze(str(f))
            mock.assert_called_once()

    def test_flac_routes_to_analyze_flac(self, tmp_path: Path) -> None:
        """analyze() should route .flac files to _analyze_flac."""
        fa = FileAnalyzer()
        f = tmp_path / "test.flac"
        f.write_bytes(b"fake")

        with patch.object(fa, "_analyze_flac", return_value=Mock()) as mock:
            fa.analyze(str(f))
            mock.assert_called_once()

    def test_wav_routes_to_analyze_wav(self, tmp_path: Path) -> None:
        """analyze() should route .wav files to _analyze_wav."""
        fa = FileAnalyzer()
        f = tmp_path / "test.wav"
        f.write_bytes(b"fake")

        with patch.object(fa, "_analyze_wav", return_value=Mock()) as mock:
            fa.analyze(str(f))
            mock.assert_called_once()

    def test_ogg_routes_to_analyze_generic(self, tmp_path: Path) -> None:
        """analyze() should route unknown extensions to _analyze_generic."""
        fa = FileAnalyzer()
        f = tmp_path / "test.ogg"
        f.write_bytes(b"fake")

        with patch.object(fa, "_analyze_generic", return_value=Mock()) as mock:
            fa.analyze(str(f))
            mock.assert_called_once()

    def test_nonexistent_file_returns_none(self) -> None:
        """analyze() should return None for a missing file."""
        fa = FileAnalyzer()
        result = fa.analyze("/nonexistent/file.mp3")
        assert result is None


class TestAnalyzeMp3:
    """Tests for MP3 metadata extraction."""

    def test_cbr_metadata(self, tmp_path: Path) -> None:
        """_analyze_mp3 should extract CBR MP3 metadata."""
        fa = FileAnalyzer()
        f = tmp_path / "test.mp3"
        f.write_bytes(b"fake")

        mock_info = Mock(
            sample_rate=44100,
            length=180.0,
            channels=2,
            bitrate=320000,
            bitrate_mode=0,
            encoder_info="LAME 3.100",
        )
        mock_audio = Mock(info=mock_info, tags=None)

        with patch("bitrater.file_analyzer.MP3", return_value=mock_audio):
            result = fa._analyze_mp3(f)

        assert result.format == "mp3"
        assert result.sample_rate == 44100
        assert result.duration == 180.0
        assert result.channels == 2
        assert result.bitrate == 320
        assert result.encoding_type == "CBR"
        assert result.encoder == "LAME 3.100"

    def test_vbr_metadata(self, tmp_path: Path) -> None:
        """_analyze_mp3 should detect VBR encoding."""
        fa = FileAnalyzer()
        f = tmp_path / "test.mp3"
        f.write_bytes(b"fake")

        mock_info = Mock(
            sample_rate=44100,
            length=200.0,
            channels=2,
            bitrate=245000,
            bitrate_mode=BitrateMode.VBR,
            encoder_info="",
        )
        mock_audio = Mock(info=mock_info, tags=Mock())
        mock_audio.tags.__contains__ = lambda self, key: key == "TSSE"
        mock_audio.tags.__getitem__ = lambda self, key: "LAME V0"

        with patch("bitrater.file_analyzer.MP3", return_value=mock_audio):
            result = fa._analyze_mp3(f)

        assert result.encoding_type == "VBR"


class TestDetermineMp3EncodingType:
    """Tests for _determine_mp3_encoding_type."""

    def test_vbr_mode(self) -> None:
        """BitrateMode.VBR should return VBR."""
        fa = FileAnalyzer()
        mock_audio = Mock(info=Mock(bitrate_mode=BitrateMode.VBR))
        assert fa._determine_mp3_encoding_type(mock_audio) == "VBR"

    def test_abr_mode(self) -> None:
        """BitrateMode.ABR should return ABR."""
        fa = FileAnalyzer()
        mock_audio = Mock(info=Mock(bitrate_mode=BitrateMode.ABR))
        assert fa._determine_mp3_encoding_type(mock_audio) == "ABR"

    def test_cbr_mode(self) -> None:
        """BitrateMode.CBR should return CBR."""
        fa = FileAnalyzer()
        mock_audio = Mock(info=Mock(bitrate_mode=BitrateMode.CBR))
        assert fa._determine_mp3_encoding_type(mock_audio) == "CBR"

    def test_missing_bitrate_mode_defaults_to_cbr(self) -> None:
        """Missing bitrate_mode attribute should default to CBR."""
        fa = FileAnalyzer()
        mock_audio = Mock(info=Mock(spec=[]))  # no bitrate_mode attr
        assert fa._determine_mp3_encoding_type(mock_audio) == "CBR"


class TestGetEncoder:
    """Tests for _get_encoder."""

    def test_encoder_info_takes_priority(self) -> None:
        """encoder_info on info object should be preferred."""
        fa = FileAnalyzer()
        mock_audio = Mock(info=Mock(encoder_info="LAME 3.100"), tags=None)
        assert fa._get_encoder(mock_audio) == "LAME 3.100"

    def test_id3_tsse_tag_fallback(self) -> None:
        """Should fall back to TSSE ID3 tag when no encoder_info."""
        fa = FileAnalyzer()
        mock_info = Mock(spec=[])  # no encoder_info
        mock_tags = {"TSSE": "LAME V0"}
        mock_audio = Mock(info=mock_info, tags=mock_tags)
        assert fa._get_encoder(mock_audio) == "LAME V0"

    def test_unknown_when_no_info(self) -> None:
        """Should return 'Unknown' when no encoder info available."""
        fa = FileAnalyzer()
        mock_info = Mock(spec=[])  # no encoder_info
        mock_audio = Mock(info=mock_info, tags=None)
        assert fa._get_encoder(mock_audio) == "Unknown"


class TestAnalyzeFlac:
    """Tests for FLAC metadata extraction."""

    def test_flac_lossless_metadata(self, tmp_path: Path) -> None:
        """_analyze_flac should extract lossless FLAC metadata."""
        fa = FileAnalyzer()
        f = tmp_path / "test.flac"
        f.write_bytes(b"fake")

        mock_info = Mock(
            sample_rate=44100,
            length=240.0,
            channels=2,
            bits_per_sample=16,
        )
        mock_tags = Mock()
        mock_tags.get.return_value = ["reference libFLAC 1.3.3"]
        mock_audio = Mock(info=mock_info, tags=mock_tags)

        with patch("bitrater.file_analyzer.FLAC", return_value=mock_audio):
            result = fa._analyze_flac(f)

        assert result.format == "flac"
        assert result.encoding_type == "lossless"
        assert result.bits_per_sample == 16
        assert result.encoder == "reference libFLAC 1.3.3"

    def test_flac_no_encoder_tag(self, tmp_path: Path) -> None:
        """_analyze_flac should default to 'Unknown' when no ENCODER tag."""
        fa = FileAnalyzer()
        f = tmp_path / "test.flac"
        f.write_bytes(b"fake")

        mock_info = Mock(sample_rate=44100, length=120.0, channels=2, bits_per_sample=16)
        mock_audio = Mock(info=mock_info, tags=None)

        with patch("bitrater.file_analyzer.FLAC", return_value=mock_audio):
            result = fa._analyze_flac(f)

        assert result.encoder == "Unknown"


class TestAnalyzeErrorHandling:
    """Tests for error handling in analyze()."""

    def test_value_error_returns_none(self, tmp_path: Path) -> None:
        """ValueError during analysis should return None."""
        fa = FileAnalyzer()
        f = tmp_path / "test.mp3"
        f.write_bytes(b"fake")

        with patch("bitrater.file_analyzer.MP3", side_effect=ValueError("corrupt")):
            result = fa.analyze(str(f))

        assert result is None

    def test_unexpected_error_returns_none(self, tmp_path: Path) -> None:
        """Unexpected exceptions should return None."""
        fa = FileAnalyzer()
        f = tmp_path / "test.mp3"
        f.write_bytes(b"fake")

        with patch("bitrater.file_analyzer.MP3", side_effect=OSError("disk error")):
            result = fa.analyze(str(f))

        assert result is None
