"""Tests for the transcode module."""

import concurrent.futures
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from bitrater.constants import CBR_BITRATES, VBR_PRESETS
from bitrater.transcode import AudioEncoder, ProgressTracker, migrate_existing_files


class TestAudioEncoderInit:
    """Tests for AudioEncoder initialization."""

    @patch("shutil.which")
    def test_init_creates_output_dir(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        AudioEncoder(source, output)
        assert output.exists()

    @patch("shutil.which")
    def test_missing_lame_raises(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: None if x == "lame" else f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        with pytest.raises(RuntimeError, match="lame"):
            AudioEncoder(source, output)

    @patch("shutil.which")
    def test_missing_flac_raises(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: None if x == "flac" else f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        with pytest.raises(RuntimeError, match="flac"):
            AudioEncoder(source, output)


class TestSanitizeFilename:
    """Tests for _sanitize_filename."""

    def test_basic_filename(self):
        assert AudioEncoder._sanitize_filename("hello_world") == "hello_world"

    def test_brackets_replaced(self):
        result = AudioEncoder._sanitize_filename("song[remix](edit){v2}")
        assert "[" not in result
        assert "]" not in result
        assert "(" not in result
        assert ")" not in result
        assert "{" not in result
        assert "}" not in result

    def test_unicode_normalized(self):
        result = AudioEncoder._sanitize_filename("café")
        assert isinstance(result, str)

    def test_special_chars_removed(self):
        result = AudioEncoder._sanitize_filename("song$name@here!")
        # Only word chars, dashes, dots should remain
        assert all(c.isalnum() or c in "_-." for c in result)

    def test_multiple_underscores_collapsed(self):
        result = AudioEncoder._sanitize_filename("a___b___c")
        assert "___" not in result
        assert result == "a_b_c"

    def test_leading_trailing_underscores_stripped(self):
        result = AudioEncoder._sanitize_filename("_test_")
        assert not result.startswith("_")
        assert not result.endswith("_")


class TestCollectSourceFiles:
    """Tests for _collect_source_files."""

    @patch("shutil.which")
    def test_finds_flac_and_wav(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        (source / "a.flac").touch()
        (source / "b.wav").touch()
        (source / "c.mp3").touch()  # Should be ignored

        encoder = AudioEncoder(source, tmp_path / "output")
        files = encoder._collect_source_files()

        extensions = {f.suffix for f in files}
        assert ".flac" in extensions
        assert ".wav" in extensions
        assert ".mp3" not in extensions

    @patch("shutil.which")
    def test_finds_nested_files(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        sub = source / "album1"
        sub.mkdir(parents=True)
        (sub / "track.flac").touch()

        encoder = AudioEncoder(source, tmp_path / "output")
        files = encoder._collect_source_files()
        assert len(files) == 1


class TestCheckOutputsExist:
    """Tests for _check_outputs_exist."""

    @patch("shutil.which")
    def test_returns_false_when_none_exist(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")
        assert encoder._check_outputs_exist("test_song") is False

    @patch("shutil.which")
    def test_returns_true_when_all_exist(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        encoder = AudioEncoder(source, output)

        # Create all expected output files in new location
        for bitrate in CBR_BITRATES:
            p = output / "lossy" / str(bitrate) / "test_song.mp3"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        for preset in VBR_PRESETS:
            p = output / "lossy" / f"v{preset}" / "test_song.mp3"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        assert encoder._check_outputs_exist("test_song") is True

    @patch("shutil.which")
    def test_returns_true_when_all_exist_old_location(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        encoder = AudioEncoder(source, output)

        # Create all expected output files in old location
        for bitrate in CBR_BITRATES:
            p = output / str(bitrate) / "test_song.mp3"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        for preset in VBR_PRESETS:
            p = output / f"v{preset}" / "test_song.mp3"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        assert encoder._check_outputs_exist("test_song") is True


class TestEnsureLosslessSymlink:
    """Tests for _ensure_lossless_symlink."""

    @patch("shutil.which")
    def test_creates_symlink(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        source_file = source / "track.flac"
        source_file.touch()
        output = tmp_path / "output"

        encoder = AudioEncoder(source, output)
        encoder._ensure_lossless_symlink(source_file, "track")

        symlink = output / "lossless" / "track.flac"
        assert symlink.is_symlink()
        assert symlink.resolve() == source_file.resolve()

    @patch("shutil.which")
    def test_does_not_overwrite_existing_symlink(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        source_file = source / "track.flac"
        source_file.touch()
        output = tmp_path / "output"

        encoder = AudioEncoder(source, output)
        encoder._ensure_lossless_symlink(source_file, "track")
        # Call again - should not raise
        encoder._ensure_lossless_symlink(source_file, "track")


class TestCreateEncodingTasks:
    """Tests for _create_encoding_tasks."""

    @patch("shutil.which")
    def test_creates_tasks_for_all_formats(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        encoder = AudioEncoder(source, output)
        wav_path = tmp_path / "test.wav"
        wav_path.touch()

        tasks = encoder._create_encoding_tasks(wav_path, "test")
        assert len(tasks) == len(CBR_BITRATES) + len(VBR_PRESETS)

    @patch("shutil.which")
    def test_skips_existing_outputs(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        encoder = AudioEncoder(source, output)

        # Pre-create one output
        existing = output / "lossy" / "128" / "test.mp3"
        existing.parent.mkdir(parents=True)
        existing.touch()

        wav_path = tmp_path / "test.wav"
        wav_path.touch()

        tasks = encoder._create_encoding_tasks(wav_path, "test")
        # Should be one fewer task
        assert len(tasks) == len(CBR_BITRATES) + len(VBR_PRESETS) - 1


class TestRunEncode:
    """Tests for _run_encode."""

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_successful_encode(self, mock_run, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")

        output_path = tmp_path / "out.mp3"
        task = (["lame", "input.wav", str(output_path)], output_path, "CBR-128")

        result = encoder._run_encode(task)
        assert result is True

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_failed_encode_cleans_up(self, mock_run, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        mock_run.side_effect = subprocess.CalledProcessError(1, "lame", stderr="error")

        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")

        output_path = tmp_path / "out.mp3"
        output_path.touch()  # Simulate partial file
        task = (["lame", "input.wav", str(output_path)], output_path, "CBR-128")

        result = encoder._run_encode(task)
        assert result is False
        assert not output_path.exists()


class TestCleanupTempFile:
    """Tests for _cleanup_temp_file."""

    @patch("shutil.which")
    def test_cleanup_temp_file(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")

        temp = tmp_path / "temp_song.wav"
        temp.touch()
        encoder._cleanup_temp_file(temp)
        assert not temp.exists()

    @patch("shutil.which")
    def test_cleanup_ignores_non_temp_files(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")

        non_temp = tmp_path / "song.wav"
        non_temp.touch()
        encoder._cleanup_temp_file(non_temp)
        assert non_temp.exists()

    @patch("shutil.which")
    def test_cleanup_none_path(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")
        # Should not raise
        encoder._cleanup_temp_file(None)


class TestCheckMigrationNeeded:
    """Tests for check_migration_needed."""

    @patch("shutil.which")
    def test_no_old_dirs(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")
        assert encoder.check_migration_needed() is False

    @patch("shutil.which")
    def test_old_dirs_with_files(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        encoder = AudioEncoder(source, output)

        # Create old-style directory with files
        old_dir = output / "128"
        old_dir.mkdir(parents=True)
        (old_dir / "test.mp3").touch()

        assert encoder.check_migration_needed() is True


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_init(self):
        tracker = ProgressTracker(10, 6, "Test")
        assert tracker.total_files == 10
        assert tracker.total_formats == 6
        assert tracker.total_tasks == 60
        assert tracker.completed_files == 0
        assert tracker.completed_tasks == 0

    def test_complete_task(self):
        tracker = ProgressTracker(10, 6)
        tracker.complete_task()
        assert tracker.completed_tasks == 1

    def test_complete_file(self):
        tracker = ProgressTracker(10, 6)
        tracker.complete_file("test.flac", 6, 6)
        assert tracker.completed_files == 1

    def test_finish(self):
        tracker = ProgressTracker(10, 6)
        tracker.completed_files = 5
        tracker.completed_tasks = 30
        # Should not raise
        tracker.finish()


class TestMigrateExistingFiles:
    """Tests for migrate_existing_files."""

    def test_no_old_dirs(self, tmp_path):
        output = tmp_path / "encoded"
        output.mkdir()
        # Should not raise
        migrate_existing_files(output)

    def test_dry_run_does_not_move(self, tmp_path):
        output = tmp_path / "encoded"
        output.mkdir()

        old_dir = output / "128"
        old_dir.mkdir()
        mp3 = old_dir / "test.mp3"
        mp3.touch()

        migrate_existing_files(output, dry_run=True)

        # File should still be in old location
        assert mp3.exists()
        assert not (output / "lossy" / "128" / "test.mp3").exists()

    def test_actual_migration_moves_files(self, tmp_path):
        output = tmp_path / "encoded"
        output.mkdir()

        old_dir = output / "128"
        old_dir.mkdir()
        mp3 = old_dir / "test.mp3"
        mp3.touch()

        migrate_existing_files(output, dry_run=False)

        # File should be in new location
        assert (output / "lossy" / "128" / "test.mp3").exists()
        # Old file should be gone
        assert not mp3.exists()
        # Old dir should be removed (empty)
        assert not old_dir.exists()

    def test_skips_existing_in_new_location(self, tmp_path):
        output = tmp_path / "encoded"
        output.mkdir()

        # Old location
        old_dir = output / "128"
        old_dir.mkdir()
        (old_dir / "test.mp3").write_text("old")

        # New location already has the file
        new_dir = output / "lossy" / "128"
        new_dir.mkdir(parents=True)
        (new_dir / "test.mp3").write_text("new")

        migrate_existing_files(output, dry_run=False)

        # New file should be untouched
        assert (new_dir / "test.mp3").read_text() == "new"

    def test_empty_old_dirs_no_files(self, tmp_path):
        output = tmp_path / "encoded"
        output.mkdir()

        # Old dir exists but has no mp3 files
        old_dir = output / "128"
        old_dir.mkdir()

        migrate_existing_files(output)

    def test_vbr_directories_migrated(self, tmp_path):
        output = tmp_path / "encoded"
        output.mkdir()

        old_dir = output / "v0"
        old_dir.mkdir()
        (old_dir / "track.mp3").touch()

        migrate_existing_files(output, dry_run=False)

        assert (output / "lossy" / "v0" / "track.mp3").exists()

    def test_old_dir_with_remaining_files(self, tmp_path):
        """Old dir with non-mp3 files should not be removed."""
        output = tmp_path / "encoded"
        output.mkdir()

        old_dir = output / "128"
        old_dir.mkdir()
        (old_dir / "test.mp3").touch()
        (old_dir / "notes.txt").touch()  # Non-mp3 file

        migrate_existing_files(output, dry_run=False)

        # Old dir should still exist (has remaining files)
        assert old_dir.exists()


class TestPrepareFile:
    """Tests for _prepare_file."""

    @patch("shutil.which")
    def test_returns_none_when_all_outputs_exist(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        encoder = AudioEncoder(source, output)

        # Pre-create all outputs
        for bitrate in CBR_BITRATES:
            p = output / "lossy" / str(bitrate) / "test.mp3"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        for preset in VBR_PRESETS:
            p = output / "lossy" / f"v{preset}" / "test.mp3"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        result = encoder._prepare_file(source / "test.flac", "test")
        assert result is None

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_decodes_flac_to_wav(self, mock_run, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        source_file = source / "track.flac"
        source_file.touch()
        output = tmp_path / "output"

        encoder = AudioEncoder(source, output)
        result = encoder._prepare_file(source_file, "track")

        assert result is not None
        assert "temp_track.wav" in str(result)
        mock_run.assert_called_once()

    @patch("shutil.which")
    def test_wav_file_used_directly(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        wav_file = source / "track.wav"
        wav_file.touch()
        output = tmp_path / "output"

        encoder = AudioEncoder(source, output)
        result = encoder._prepare_file(wav_file, "track")

        # WAV files are used directly without decoding
        assert result == wav_file


class TestDecodeFlac:
    """Tests for _decode_flac."""

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_successful_decode(self, mock_run, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")
        encoder._decode_flac(source / "test.flac", tmp_path / "test.wav")

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "--decode" in call_args

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_failed_decode_raises(self, mock_run, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        mock_run.side_effect = subprocess.CalledProcessError(1, "flac", stderr="decode error")
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")
        with pytest.raises(subprocess.CalledProcessError):
            encoder._decode_flac(source / "test.flac", tmp_path / "test.wav")


class TestRunEncodeExceptions:
    """Tests for _run_encode exception handling."""

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_generic_exception_cleans_up(self, mock_run, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        mock_run.side_effect = OSError("disk full")
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")
        output_path = tmp_path / "out.mp3"
        output_path.touch()
        task = (["lame", "in.wav", str(output_path)], output_path, "CBR-128")

        result = encoder._run_encode(task)
        assert result is False
        assert not output_path.exists()


class TestDrainCompletedFile:
    """Tests for _drain_completed_file."""

    @patch("shutil.which")
    def test_drains_successful_futures(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")
        progress = ProgressTracker(1, 2)

        # Create completed futures
        f1 = concurrent.futures.Future()
        f1.set_result(True)
        f2 = concurrent.futures.Future()
        f2.set_result(True)

        temp_wav = tmp_path / "temp_test.wav"
        temp_wav.touch()

        in_flight = [(temp_wav, [f1, f2], "test.flac", 2)]
        encoder._drain_completed_file(in_flight, progress)

        assert len(in_flight) == 0
        assert progress.completed_files == 1
        assert not temp_wav.exists()

    @patch("shutil.which")
    def test_drains_failed_futures(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")
        progress = ProgressTracker(1, 2)

        f1 = concurrent.futures.Future()
        f1.set_result(True)
        f2 = concurrent.futures.Future()
        f2.set_exception(RuntimeError("encode failed"))

        in_flight = [(None, [f1, f2], "test.flac", 2)]
        encoder._drain_completed_file(in_flight, progress)

        assert len(in_flight) == 0
        assert progress.completed_files == 1


class TestDrainReadyFiles:
    """Tests for _drain_ready_files."""

    @patch("shutil.which")
    def test_drains_completed_leaves_pending(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()

        encoder = AudioEncoder(source, tmp_path / "output")
        progress = ProgressTracker(2, 1)

        # First file: completed
        f1 = concurrent.futures.Future()
        f1.set_result(True)

        # Second file: still running
        f2 = concurrent.futures.Future()  # Not completed

        in_flight = [
            (None, [f1], "done.flac", 1),
            (None, [f2], "running.flac", 1),
        ]
        encoder._drain_ready_files(in_flight, progress)

        assert len(in_flight) == 1
        assert in_flight[0][2] == "running.flac"


class TestProcessFiles:
    """Tests for process_files."""

    @patch("shutil.which")
    def test_process_files_empty_source_raises(self, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()  # Empty directory

        encoder = AudioEncoder(source, tmp_path / "output")
        with pytest.raises(ValueError, match="No source files"):
            encoder.process_files()

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_process_files_skips_already_encoded(self, mock_run, mock_which, tmp_path):
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        source = tmp_path / "source"
        source.mkdir()
        (source / "track.flac").touch()
        output = tmp_path / "output"

        # Pre-create all outputs so nothing needs encoding
        for bitrate in CBR_BITRATES:
            p = output / "lossy" / str(bitrate) / "track.mp3"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        for preset in VBR_PRESETS:
            p = output / "lossy" / f"v{preset}" / "track.mp3"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        encoder = AudioEncoder(source, output)
        encoder.process_files(max_workers=1)

        # No encoding subprocess calls (only the flac decode may be attempted)
        # but since outputs exist, _prepare_file returns None
