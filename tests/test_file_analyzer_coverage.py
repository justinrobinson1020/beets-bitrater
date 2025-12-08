"""Additional tests for file_analyzer.py to improve coverage."""

from unittest.mock import MagicMock, patch

from mutagen.mp3 import BitrateMode

from bitrater.file_analyzer import FileAnalyzer


class TestAnalyzeWav:
    """Tests for _analyze_wav."""

    @patch("bitrater.file_analyzer.WAVE")
    def test_wav_with_bits_per_sample(self, mock_wave_cls, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.touch()

        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 44100
        mock_audio.info.length = 60.0
        mock_audio.info.channels = 2
        mock_audio.info.bits_per_sample = 24
        mock_wave_cls.return_value = mock_audio

        analyzer = FileAnalyzer()
        result = analyzer._analyze_wav(wav_file)

        assert result is not None
        assert result.format == "wav"
        assert result.encoding_type == "lossless"
        assert result.bits_per_sample == 24
        assert result.encoder == "PCM"

    @patch("bitrater.file_analyzer.WAVE")
    def test_wav_without_bits_per_sample(self, mock_wave_cls, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.touch()

        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 44100
        mock_audio.info.length = 60.0
        mock_audio.info.channels = 2
        mock_audio.info.bits_per_sample = None
        # Remove bits_per_sample attribute entirely
        del mock_audio.info.bits_per_sample
        mock_wave_cls.return_value = mock_audio

        analyzer = FileAnalyzer()
        result = analyzer._analyze_wav(wav_file)

        assert result is not None
        assert result.bits_per_sample == 16  # Default


class TestAnalyzeGeneric:
    """Tests for _analyze_generic."""

    @patch("bitrater.file_analyzer.MutagenFile")
    def test_generic_returns_none_for_unreadable(self, mock_mutagen, tmp_path):
        test_file = tmp_path / "test.ogg"
        test_file.touch()

        mock_mutagen.return_value = None

        analyzer = FileAnalyzer()
        result = analyzer._analyze_generic(test_file, "ogg")
        assert result is None

    @patch("bitrater.file_analyzer.MutagenFile")
    def test_generic_lossy_format(self, mock_mutagen, tmp_path):
        test_file = tmp_path / "test.ogg"
        test_file.touch()

        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 44100
        mock_audio.info.length = 120.0
        mock_audio.info.channels = 2
        mock_audio.info.bitrate = 192000
        mock_mutagen.return_value = mock_audio

        analyzer = FileAnalyzer()
        result = analyzer._analyze_generic(test_file, "ogg")

        assert result is not None
        assert result.format == "ogg"
        assert result.encoding_type == "lossy"
        assert result.bitrate == 192

    @patch("bitrater.file_analyzer.MutagenFile")
    def test_generic_lossless_format(self, mock_mutagen, tmp_path):
        test_file = tmp_path / "test.alac"
        test_file.touch()

        mock_audio = MagicMock()
        mock_audio.info.sample_rate = 44100
        mock_audio.info.length = 120.0
        mock_audio.info.channels = 2
        mock_audio.info.bitrate = None
        del mock_audio.info.bitrate
        mock_mutagen.return_value = mock_audio

        analyzer = FileAnalyzer()
        result = analyzer._analyze_generic(test_file, "alac")

        assert result is not None
        assert result.encoding_type == "lossless"
        assert result.bitrate is None


class TestAnalyzeErrors:
    """Tests for error handling in analyze."""

    def test_nonexistent_file(self):
        analyzer = FileAnalyzer()
        result = analyzer.analyze("/nonexistent/file.mp3")
        assert result is None

    @patch("bitrater.file_analyzer.MP3")
    def test_unexpected_error_returns_none(self, mock_mp3, tmp_path):
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        mock_mp3.side_effect = OSError("disk error")

        analyzer = FileAnalyzer()
        result = analyzer.analyze(str(test_file))
        assert result is None

    @patch("bitrater.file_analyzer.MP3")
    def test_value_error_returns_none(self, mock_mp3, tmp_path):
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        mock_mp3.side_effect = ValueError("invalid format")

        analyzer = FileAnalyzer()
        result = analyzer.analyze(str(test_file))
        assert result is None


class TestDetermineMp3EncodingType:
    """Tests for _determine_mp3_encoding_type."""

    def test_vbr_mode(self):
        analyzer = FileAnalyzer()
        mock_audio = MagicMock()
        mock_audio.info.bitrate_mode = BitrateMode.VBR
        assert analyzer._determine_mp3_encoding_type(mock_audio) == "VBR"

    def test_abr_mode(self):
        analyzer = FileAnalyzer()
        mock_audio = MagicMock()
        mock_audio.info.bitrate_mode = BitrateMode.ABR
        assert analyzer._determine_mp3_encoding_type(mock_audio) == "ABR"

    def test_cbr_mode(self):
        analyzer = FileAnalyzer()
        mock_audio = MagicMock()
        mock_audio.info.bitrate_mode = BitrateMode.CBR
        assert analyzer._determine_mp3_encoding_type(mock_audio) == "CBR"

    def test_unknown_mode_defaults_to_cbr(self):
        analyzer = FileAnalyzer()
        mock_audio = MagicMock()
        mock_audio.info.bitrate_mode = BitrateMode.UNKNOWN
        assert analyzer._determine_mp3_encoding_type(mock_audio) == "CBR"

    def test_no_bitrate_mode_attribute(self):
        analyzer = FileAnalyzer()
        mock_audio = MagicMock()
        del mock_audio.info.bitrate_mode
        assert analyzer._determine_mp3_encoding_type(mock_audio) == "CBR"


class TestGetEncoder:
    """Tests for _get_encoder."""

    def test_encoder_from_info(self):
        analyzer = FileAnalyzer()
        mock_audio = MagicMock()
        mock_audio.info.encoder_info = "LAME3.100"
        assert analyzer._get_encoder(mock_audio) == "LAME3.100"

    def test_encoder_from_tags(self):
        analyzer = FileAnalyzer()
        mock_audio = MagicMock()
        mock_audio.info.encoder_info = ""
        mock_audio.tags = {"TSSE": "LAME v3.99"}
        assert analyzer._get_encoder(mock_audio) == "LAME v3.99"

    def test_unknown_encoder(self):
        analyzer = FileAnalyzer()
        mock_audio = MagicMock()
        mock_audio.info.encoder_info = ""
        mock_audio.tags = {}
        assert analyzer._get_encoder(mock_audio) == "Unknown"

    def test_no_tags_no_encoder_info(self):
        analyzer = FileAnalyzer()
        mock_audio = MagicMock()
        mock_audio.info.encoder_info = ""
        mock_audio.tags = None
        assert analyzer._get_encoder(mock_audio) == "Unknown"
