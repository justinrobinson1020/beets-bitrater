"""Tests for parallel training with dynamic worker count."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from beetsplug.bitrater.analyzer import AudioQualityAnalyzer


class TestDynamicWorkerCount:
    """Tests for dynamic worker count based on CPU cores."""

    def test_get_default_workers_exists(self) -> None:
        """Analyzer should have _get_default_workers method."""
        analyzer = AudioQualityAnalyzer()
        assert hasattr(analyzer, "_get_default_workers")

    def test_default_workers_uses_50_percent_of_cpu_count(self) -> None:
        """Default workers should be 50% of os.cpu_count() (conservative to prevent overload)."""
        analyzer = AudioQualityAnalyzer()

        with patch("os.cpu_count", return_value=10):
            default_workers = analyzer._get_default_workers()
            # 50% of 10 = 5
            assert default_workers == 5

        with patch("os.cpu_count", return_value=16):
            default_workers = analyzer._get_default_workers()
            # 50% of 16 = 8
            assert default_workers == 8

        with patch("os.cpu_count", return_value=4):
            default_workers = analyzer._get_default_workers()
            # 50% of 4 = 2
            assert default_workers == 2

    def test_default_workers_minimum_one(self) -> None:
        """Default workers should be at least 1 even if cpu_count returns None."""
        analyzer = AudioQualityAnalyzer()

        with patch("os.cpu_count", return_value=None):
            default_workers = analyzer._get_default_workers()
            assert default_workers >= 1

    def test_default_workers_scales_with_cpu_count(self) -> None:
        """Default workers should scale with available CPU cores at 50%."""
        analyzer = AudioQualityAnalyzer()

        # Test with different CPU counts - should be 50% of each
        with patch("os.cpu_count", return_value=4):
            workers_4 = analyzer._get_default_workers()
            assert workers_4 == 2  # 50% of 4 = 2

        with patch("os.cpu_count", return_value=8):
            workers_8 = analyzer._get_default_workers()
            assert workers_8 == 4  # 50% of 8 = 4

        # More CPUs should mean more workers
        assert workers_8 > workers_4

    def test_default_workers_not_hardcoded(self) -> None:
        """Verify workers are dynamically determined, not hardcoded."""
        analyzer = AudioQualityAnalyzer()

        # Patch to return a different value
        with patch("os.cpu_count", return_value=1):
            single_cpu_workers = analyzer._get_default_workers()

        with patch("os.cpu_count", return_value=16):
            many_cpu_workers = analyzer._get_default_workers()

        # With 1 vs 16 CPUs, we should see different worker counts
        assert single_cpu_workers != many_cpu_workers, "Worker count should scale with CPU count"  # At minimum different from hardcoded


class TestParallelFeatureExtraction:
    """Tests for parallel feature extraction during training."""

    def test_train_parallel_method_exists(self) -> None:
        """Training should have a parallel processing method."""
        analyzer = AudioQualityAnalyzer()
        assert hasattr(analyzer, "train_parallel")

    def test_train_parallel_accepts_num_workers(self) -> None:
        """train_parallel should accept num_workers parameter."""
        analyzer = AudioQualityAnalyzer()

        # Should accept num_workers parameter without error
        # Testing with empty data should raise ValueError (not TypeError)
        with pytest.raises(ValueError, match="No training data"):
            analyzer.train_parallel(
                training_data={},
                num_workers=2,
            )

    def test_train_from_directory_uses_parallel(self, tmp_path, sample_features) -> None:
        """train_from_directory should use parallel processing by default."""
        analyzer = AudioQualityAnalyzer()

        # Create directory structure
        for bitrate in ["128", "320"]:
            (tmp_path / bitrate).mkdir()
            for i in range(3):
                (tmp_path / bitrate / f"test_{i}.mp3").write_bytes(b"fake")

        (tmp_path / "lossless").mkdir()
        for i in range(3):
            (tmp_path / "lossless" / f"test_{i}.flac").write_bytes(b"fake")

        with patch.object(analyzer, "train_parallel") as mock_train_parallel, \
             patch.object(analyzer.spectrum_analyzer, "analyze_file", return_value=sample_features), \
             patch.object(analyzer.file_analyzer, "analyze", return_value=None):
            analyzer.train_from_directory(tmp_path)
            assert mock_train_parallel.called


class TestExtractFeaturesWorker:
    """Tests for the _extract_features_worker function."""

    @pytest.fixture(autouse=True)
    def reset_worker_state(self) -> None:
        """Reset worker globals before each test."""
        import beetsplug.bitrater.analyzer as analyzer_module
        # Reset the worker singleton state
        analyzer_module._worker_initialized = False
        analyzer_module._worker_analyzer = None
        analyzer_module._worker_file_analyzer = None
        yield
        # Clean up after test
        analyzer_module._worker_initialized = False
        analyzer_module._worker_analyzer = None
        analyzer_module._worker_file_analyzer = None

    def test_extract_features_worker_returns_tuple(self, sample_features: SpectralFeatures, tmp_path) -> None:
        """Worker function should return (file_path, features) tuple for valid file."""
        from beetsplug.bitrater.analyzer import _extract_features_worker

        # Create a real audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        # Mock the SpectrumAnalyzer to return features
        # Patch at the source module since worker uses lazy import
        with patch("beetsplug.bitrater.spectrum.SpectrumAnalyzer") as mock_spectrum:
            mock_instance = mock_spectrum.return_value
            mock_instance.analyze_file.return_value = sample_features

            result = _extract_features_worker(str(audio_file))

        assert isinstance(result, tuple)
        assert len(result) == 2
        file_path, features = result
        assert file_path == str(audio_file)
        assert features is not None
        assert features == sample_features

    def test_extract_features_worker_handles_missing_file(self) -> None:
        """Worker function should return (file_path, None) for missing file."""
        from beetsplug.bitrater.analyzer import _extract_features_worker

        result = _extract_features_worker("/nonexistent/file.mp3")

        assert isinstance(result, tuple)
        assert len(result) == 2
        file_path, features = result
        assert file_path == "/nonexistent/file.mp3"
        assert features is None


class TestJoblibIntegration:
    """Tests for joblib.Parallel integration in train_parallel."""

    def test_plugin_analyze_uses_threading_backend(self) -> None:
        """Plugin _analyze_items should use joblib.Parallel with threading backend."""
        from unittest.mock import MagicMock

        from beetsplug.bitrater.plugin import BitraterPlugin

        # Mock joblib.Parallel to capture the call
        with patch("beetsplug.bitrater.plugin.Parallel") as mock_parallel:
            mock_parallel.return_value = MagicMock(return_value=[None, None])

            plugin = BitraterPlugin()

            # Create mock items with paths
            mock_item1 = MagicMock()
            mock_item1.path = "/fake/path1.mp3"
            mock_item2 = MagicMock()
            mock_item2.path = "/fake/path2.mp3"

            plugin._analyze_items([mock_item1, mock_item2], thread_count=4)

            # Verify Parallel was called with threading backend
            mock_parallel.assert_called()
            call_kwargs = mock_parallel.call_args[1]
            assert call_kwargs.get("backend") == "threading", "plugin should use threading backend"
            assert call_kwargs.get("n_jobs") == 4, "n_jobs should match thread_count"

    def test_train_parallel_uses_loky_backend(self, tmp_path) -> None:
        """train_parallel should use joblib.Parallel with loky backend."""
        from unittest.mock import MagicMock

        # Create minimal files
        files = {}
        for i in range(2):
            f = tmp_path / f"test_{i}.mp3"
            f.write_bytes(b"fake audio")
            files[str(f)] = i % 7

        # Mock parallel_config and Parallel to verify configuration
        with patch("beetsplug.bitrater.analyzer.parallel_config") as mock_config:
            with patch("beetsplug.bitrater.analyzer.Parallel") as mock_parallel:
                # Make Parallel return an empty list to avoid further processing
                mock_parallel.return_value = MagicMock(return_value=[])
                mock_config.return_value.__enter__ = MagicMock()
                mock_config.return_value.__exit__ = MagicMock()

                analyzer = AudioQualityAnalyzer()
                try:
                    analyzer.train_parallel(files, num_workers=2)
                except (ValueError, StopIteration):
                    pass  # Expected - empty results from mock

                # Verify parallel_config was called with loky backend
                mock_config.assert_called()
                config_kwargs = mock_config.call_args[1]
                assert config_kwargs.get("backend") == "loky", "train_parallel should use loky backend"
                assert config_kwargs.get("inner_max_num_threads") == 1, "Should limit inner threads"

                # Verify Parallel n_jobs matches num_workers
                mock_parallel.assert_called()
                call_kwargs = mock_parallel.call_args[1]
                assert call_kwargs.get("n_jobs") == 2, "n_jobs should match num_workers"

    def test_train_parallel_returns_correct_structure(self, tmp_path, sample_features) -> None:
        """train_parallel should return results with correct structure."""
        analyzer = AudioQualityAnalyzer()

        # Create just 2 files for quick test
        files = {}
        for i in range(2):
            f = tmp_path / f"test_{i}.mp3"
            f.write_bytes(b"fake audio content")
            files[str(f)] = i % 7

        # Mock Parallel to return controlled feature tuples (bypasses loky subprocess)
        mock_results = [(path, sample_features) for path in files.keys()]
        with patch("beetsplug.bitrater.analyzer.Parallel") as mock_parallel_cls:
            mock_parallel_cls.return_value = Mock(return_value=mock_results)
            result = analyzer.train_parallel(files, num_workers=1)

        # Verify structure of result
        assert isinstance(result, dict)
        assert "total_files" in result
        assert "successful" in result
        assert "failed" in result
        assert "extraction_time" in result
        assert "workers" in result
        assert "throughput" in result

        # Verify counts
        assert result["total_files"] == 2
        assert result["successful"] + result["failed"] == 2
