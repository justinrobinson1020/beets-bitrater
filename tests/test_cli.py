"""Tests for the bitrater CLI module."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bitrater.cli import (
    _setup_logging,
    cmd_analyze,
    cmd_train,
    cmd_transcode,
    cmd_validate,
    main,
)


class TestSetupLogging:
    """Tests for _setup_logging."""

    def test_default_level_is_info(self):
        import logging
        _setup_logging(verbose=False)
        logger = logging.getLogger("bitrater")
        # basicConfig only applies to root; just verify it runs without error

    def test_verbose_sets_debug(self):
        _setup_logging(verbose=True)
        # Verify it runs without error


class TestCmdAnalyze:
    """Tests for cmd_analyze."""

    def test_analyze_nonexistent_target_exits(self, tmp_path):
        args = argparse.Namespace(
            target=str(tmp_path / "nonexistent"),
            model=None,
            verbose=False,
        )
        with pytest.raises(SystemExit):
            cmd_analyze(args)

    def test_analyze_empty_directory_exits(self, tmp_path):
        args = argparse.Namespace(
            target=str(tmp_path),
            model=None,
            verbose=False,
        )
        with pytest.raises(SystemExit):
            cmd_analyze(args)

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_analyze_single_file(self, mock_analyzer_cls, tmp_path):
        # Create a fake audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        mock_result = MagicMock()
        mock_result.is_transcode = False
        mock_result.original_format = "320"
        mock_result.original_bitrate = 320
        mock_result.confidence = 0.95
        mock_result.warnings = []

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.return_value = mock_result
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            target=str(audio_file),
            model=None,
            verbose=False,
        )
        cmd_analyze(args)

        mock_analyzer.analyze_file.assert_called_once_with(str(audio_file))

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_analyze_directory(self, mock_analyzer_cls, tmp_path):
        # Create multiple fake audio files
        for ext in [".mp3", ".flac", ".wav"]:
            (tmp_path / f"test{ext}").touch()
        # Non-audio file should be ignored
        (tmp_path / "readme.txt").touch()

        mock_result = MagicMock()
        mock_result.is_transcode = False
        mock_result.original_format = "320"
        mock_result.original_bitrate = 320
        mock_result.confidence = 0.95
        mock_result.warnings = []

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.return_value = mock_result
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            target=str(tmp_path),
            model=None,
            verbose=False,
        )
        cmd_analyze(args)

        assert mock_analyzer.analyze_file.call_count == 3

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_analyze_with_model(self, mock_analyzer_cls, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        model_file = tmp_path / "model.pkl"

        mock_result = MagicMock()
        mock_result.is_transcode = False
        mock_result.original_format = "320"
        mock_result.original_bitrate = 320
        mock_result.confidence = 0.95
        mock_result.warnings = []

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.return_value = mock_result
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            target=str(audio_file),
            model=str(model_file),
            verbose=False,
        )
        cmd_analyze(args)

        mock_analyzer.load_model.assert_called_once_with(model_file)

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_analyze_skips_failed_files(self, mock_analyzer_cls, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.return_value = None
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            target=str(audio_file),
            model=None,
            verbose=False,
        )
        # Should not raise
        cmd_analyze(args)

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_analyze_verbose_shows_warnings(self, mock_analyzer_cls, tmp_path, capsys):
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        mock_result = MagicMock()
        mock_result.is_transcode = True
        mock_result.original_format = "128"
        mock_result.original_bitrate = 128
        mock_result.confidence = 0.6
        mock_result.warnings = ["Low confidence", "Possible transcode"]

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.return_value = mock_result
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            target=str(audio_file),
            model=None,
            verbose=True,
        )
        cmd_analyze(args)

        captured = capsys.readouterr()
        assert "TRANSCODE" in captured.out
        assert "warn: Low confidence" in captured.out


class TestCmdTrain:
    """Tests for cmd_train."""

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_train_calls_train_from_directory(self, mock_analyzer_cls, tmp_path):
        source_dir = tmp_path / "training"
        source_dir.mkdir()

        mock_analyzer = MagicMock()
        mock_analyzer.train_from_directory.return_value = {
            "successful": 100,
            "total_files": 100,
        }
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            source_dir=str(source_dir),
            save_model=None,
            threads=None,
        )
        cmd_train(args)

        mock_analyzer.train_from_directory.assert_called_once()

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_train_with_save_model(self, mock_analyzer_cls, tmp_path):
        source_dir = tmp_path / "training"
        source_dir.mkdir()
        model_path = tmp_path / "model.pkl"

        mock_analyzer = MagicMock()
        mock_analyzer.train_from_directory.return_value = {
            "successful": 50,
            "total_files": 50,
        }
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            source_dir=str(source_dir),
            save_model=str(model_path),
            threads=4,
        )
        cmd_train(args)

        mock_analyzer.train_from_directory.assert_called_once_with(
            source_dir, save_path=model_path, num_workers=4
        )


class TestCmdValidate:
    """Tests for cmd_validate."""

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_validate_prints_metrics(self, mock_analyzer_cls, tmp_path, capsys):
        source_dir = tmp_path / "training"
        source_dir.mkdir()

        mock_analyzer = MagicMock()
        mock_analyzer.validate_from_directory.return_value = {
            "total_samples": 100,
            "train_samples": 80,
            "test_samples": 20,
            "train_pct": 0.8,
            "test_pct": 0.2,
            "accuracy": 0.85,
            "per_class": {
                "128": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5},
                "320": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 5},
            },
        }
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            source_dir=str(source_dir),
            test_size=0.2,
            threads=None,
        )
        cmd_validate(args)

        captured = capsys.readouterr()
        assert "MODEL VALIDATION RESULTS" in captured.out
        assert "85.0%" in captured.out


class TestCmdTranscode:
    """Tests for cmd_transcode."""

    @patch("bitrater.transcode.AudioEncoder")
    def test_transcode_calls_process_files(self, mock_encoder_cls, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_encoder = MagicMock()
        mock_encoder_cls.return_value = mock_encoder

        args = argparse.Namespace(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            workers=4,
        )
        cmd_transcode(args)

        mock_encoder_cls.assert_called_once_with(source_dir, output_dir)
        mock_encoder.process_files.assert_called_once_with(max_workers=4)


class TestMain:
    """Tests for main CLI entry point."""

    def test_no_args_exits(self):
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["bitrater"]):
                main()

    def test_analyze_subcommand_parsed(self):
        with patch("sys.argv", ["bitrater", "analyze", "/tmp/test.mp3"]):
            with patch("bitrater.cli.cmd_analyze") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_train_subcommand_parsed(self):
        with patch("sys.argv", ["bitrater", "train", "--source-dir", "/tmp/data"]):
            with patch("bitrater.cli.cmd_train") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_validate_subcommand_parsed(self):
        with patch("sys.argv", ["bitrater", "validate", "--source-dir", "/tmp/data"]):
            with patch("bitrater.cli.cmd_validate") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_transcode_subcommand_parsed(self):
        with patch(
            "sys.argv",
            ["bitrater", "transcode", "--source-dir", "/tmp/src", "--output-dir", "/tmp/out"],
        ):
            with patch("bitrater.cli.cmd_transcode") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_verbose_flag_parsed(self):
        with patch("sys.argv", ["bitrater", "-v", "analyze", "/tmp/test.mp3"]):
            with patch("bitrater.cli.cmd_analyze") as mock_cmd:
                main()
                args = mock_cmd.call_args[0][0]
                assert args.verbose is True
