"""Audio file metadata extraction."""

import logging
from pathlib import Path

from mutagen import File as MutagenFile
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

from .types import FileMetadata

logger = logging.getLogger(__name__)


class FileAnalyzer:
    """Extracts metadata from audio files."""

    def analyze(self, file_path: str) -> FileMetadata | None:
        """
        Extract metadata from an audio file.

        Args:
            file_path: Path to audio file

        Returns:
            FileMetadata with file information, or None if extraction fails
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        file_format = path.suffix.lower().lstrip(".")

        try:
            if file_format == "mp3":
                return self._analyze_mp3(path)
            elif file_format == "flac":
                return self._analyze_flac(path)
            elif file_format == "wav":
                return self._analyze_wav(path)
            else:
                # Try generic mutagen for other formats
                return self._analyze_generic(path, file_format)

        except Exception as e:
            logger.error(f"Error analyzing metadata for {file_path}: {e}")
            return None

    def _analyze_mp3(self, path: Path) -> FileMetadata:
        """Analyze MP3 file metadata."""
        audio = MP3(path)

        # Determine encoding type
        encoding_type = self._determine_mp3_encoding_type(audio)

        # Get encoder info
        encoder = self._get_encoder(audio)

        return FileMetadata(
            format="mp3",
            sample_rate=audio.info.sample_rate,
            duration=audio.info.length,
            channels=audio.info.channels,
            encoding_type=encoding_type,
            encoder=encoder,
            bitrate=audio.info.bitrate // 1000,  # Convert to kbps
            filesize=path.stat().st_size,
        )

    def _analyze_flac(self, path: Path) -> FileMetadata:
        """Analyze FLAC file metadata."""
        audio = FLAC(path)

        encoder = "Unknown"
        if audio.tags:
            encoder = audio.tags.get("ENCODER", ["Unknown"])[0]

        return FileMetadata(
            format="flac",
            sample_rate=audio.info.sample_rate,
            duration=audio.info.length,
            channels=audio.info.channels,
            encoding_type="lossless",
            encoder=encoder,
            bits_per_sample=audio.info.bits_per_sample,
            filesize=path.stat().st_size,
        )

    def _analyze_wav(self, path: Path) -> FileMetadata:
        """Analyze WAV file metadata."""
        audio = WAVE(path)

        bits_per_sample = 16  # Default
        if hasattr(audio.info, "bits_per_sample"):
            bits_per_sample = audio.info.bits_per_sample

        return FileMetadata(
            format="wav",
            sample_rate=audio.info.sample_rate,
            duration=audio.info.length,
            channels=audio.info.channels,
            encoding_type="lossless",
            encoder="PCM",
            bits_per_sample=bits_per_sample,
            filesize=path.stat().st_size,
        )

    def _analyze_generic(self, path: Path, file_format: str) -> FileMetadata | None:
        """Analyze other audio formats using generic mutagen."""
        audio = MutagenFile(path)
        if audio is None:
            logger.warning(f"Could not read audio file: {path}")
            return None

        info = audio.info

        # Determine if lossless based on format
        lossless_formats = {"flac", "wav", "alac", "ape", "wv", "aiff"}
        encoding_type = "lossless" if file_format in lossless_formats else "lossy"

        bitrate = None
        if hasattr(info, "bitrate") and info.bitrate:
            bitrate = info.bitrate // 1000

        return FileMetadata(
            format=file_format,
            sample_rate=getattr(info, "sample_rate", 44100),
            duration=getattr(info, "length", 0.0),
            channels=getattr(info, "channels", 2),
            encoding_type=encoding_type,
            encoder="Unknown",
            bitrate=bitrate,
            filesize=path.stat().st_size,
        )

    def _determine_mp3_encoding_type(self, audio: MP3) -> str:
        """Determine MP3 encoding type (CBR, VBR, or ABR)."""
        # Check for VBR indicators
        if hasattr(audio.info, "bitrate_mode"):
            mode = audio.info.bitrate_mode
            if mode == 1:  # VBR
                return "VBR"
            elif mode == 2:  # ABR
                return "ABR"

        # Default to CBR
        return "CBR"

    def _get_encoder(self, audio: MP3) -> str:
        """Get encoder information from MP3."""
        # Check for LAME header
        if hasattr(audio.info, "encoder_info") and audio.info.encoder_info:
            return audio.info.encoder_info

        # Check ID3 tags
        if audio.tags:
            for tag in ["TSSE", "encoder", "ENCODER"]:
                if tag in audio.tags:
                    return str(audio.tags[tag])

        return "Unknown"
