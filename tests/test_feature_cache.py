"""Tests for the feature cache module."""

import json
import time

import numpy as np
import pytest

from bitrater.feature_cache import FeatureCache


@pytest.fixture
def cache(tmp_path):
    """Create a FeatureCache instance with a temp directory."""
    c = FeatureCache(tmp_path / "cache")
    yield c
    c.worker_running = False
    if c.worker_thread.is_alive():
        c.worker_thread.join(timeout=2.0)


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a fake audio file for cache key generation."""
    f = tmp_path / "test.mp3"
    f.write_bytes(b"\x00" * 1024)
    return f


class TestFeatureCacheInit:
    """Tests for FeatureCache initialization."""

    def test_creates_directories(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = FeatureCache(cache_dir)
        assert cache.features_dir.exists()
        cache.worker_running = False
        cache.worker_thread.join(timeout=1.0)

    def test_starts_worker_thread(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        assert cache.worker_thread.is_alive()
        cache.worker_running = False
        cache.worker_thread.join(timeout=1.0)

    def test_loads_existing_metadata(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        features_dir = cache_dir / "features"
        features_dir.mkdir()
        meta_path = cache_dir / "cache_metadata.json"
        meta_path.write_text(json.dumps({"abc123": {"file_path": "/test"}}))

        cache = FeatureCache(cache_dir)
        assert "abc123" in cache.metadata
        cache.worker_running = False
        cache.worker_thread.join(timeout=1.0)


class TestGetFileKey:
    """Tests for _get_file_key."""

    def test_returns_hex_hash(self, cache, sample_audio_file):
        key = cache._get_file_key(sample_audio_file)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex length

    def test_same_file_same_key(self, cache, sample_audio_file):
        key1 = cache._get_file_key(sample_audio_file)
        key2 = cache._get_file_key(sample_audio_file)
        assert key1 == key2

    def test_different_content_different_key(self, cache, tmp_path):
        f1 = tmp_path / "a.mp3"
        f1.write_bytes(b"\x00" * 100)
        f2 = tmp_path / "b.mp3"
        f2.write_bytes(b"\xff" * 200)

        key1 = cache._get_file_key(f1)
        key2 = cache._get_file_key(f2)
        assert key1 != key2


class TestSaveAndGetFeatures:
    """Tests for save_features and get_features."""

    def test_save_and_retrieve(self, cache, sample_audio_file):
        features = np.random.rand(181).astype(np.float32)
        metadata = {"n_bands": 150, "approach": "test"}

        cache.save_features(sample_audio_file, features, metadata)

        # Wait for metadata worker to process
        cache.update_queue.join()
        time.sleep(0.1)

        result = cache.get_features(sample_audio_file)
        assert result is not None
        cached_features, cached_metadata = result
        np.testing.assert_array_almost_equal(cached_features, features)
        assert cached_metadata["n_bands"] == 150

    def test_get_nonexistent_returns_none(self, cache, sample_audio_file):
        result = cache.get_features(sample_audio_file)
        assert result is None

    def test_get_stale_cache_returns_none(self, cache, tmp_path):
        """If file is newer than cache, return None."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 100)

        features = np.random.rand(181).astype(np.float32)
        metadata = {"n_bands": 150}

        cache.save_features(audio_file, features, metadata)
        cache.update_queue.join()
        time.sleep(0.1)

        # Manually set cached_date to the past
        file_hash = cache._get_file_key(audio_file)
        with cache.metadata_lock:
            cache.metadata[file_hash]["cached_date"] = "2000-01-01T00:00:00"
        cache._save_metadata()

        # Touch the file to make it newer
        time.sleep(0.01)
        audio_file.write_bytes(b"\xff" * 100)

        result = cache.get_features(audio_file)
        assert result is None


class TestClearCache:
    """Tests for clear_cache."""

    def test_clear_removes_files(self, cache, sample_audio_file):
        features = np.random.rand(181).astype(np.float32)
        cache.save_features(sample_audio_file, features, {"n_bands": 150})
        cache.update_queue.join()
        time.sleep(0.1)

        # Verify file exists
        npz_files = list(cache.features_dir.glob("*.npz"))
        assert len(npz_files) > 0

        cache.clear_cache()
        cache.update_queue.join()
        time.sleep(0.1)

        npz_files = list(cache.features_dir.glob("*.npz"))
        assert len(npz_files) == 0

    def test_clear_empties_metadata(self, cache, sample_audio_file):
        features = np.random.rand(181).astype(np.float32)
        cache.save_features(sample_audio_file, features, {"n_bands": 150})
        cache.update_queue.join()
        time.sleep(0.1)

        cache.clear_cache()
        cache.update_queue.join()
        time.sleep(0.1)

        assert cache.metadata == {}


class TestGetCacheInfo:
    """Tests for get_cache_info."""

    def test_empty_cache_info(self, cache):
        info = cache.get_cache_info()
        assert info["total_files"] == 0
        assert info["total_size_mb"] == 0.0

    def test_cache_info_with_files(self, cache, sample_audio_file):
        features = np.random.rand(181).astype(np.float32)
        cache.save_features(sample_audio_file, features, {"n_bands": 150})
        cache.update_queue.join()
        time.sleep(0.1)

        info = cache.get_cache_info()
        assert info["total_files"] == 1
        assert info["total_size_mb"] > 0
        assert info["last_modified"] is not None


class TestFileLock:
    """Tests for _file_lock context manager."""

    def test_metadata_lock_path_reuses_handle(self, cache):
        lock_path = cache._metadata_lock_path
        with cache._file_lock(lock_path):
            # First call opens the file
            assert cache._metadata_lock_file is not None
            first_handle = cache._metadata_lock_file

        with cache._file_lock(lock_path):
            # Second call reuses the handle
            assert cache._metadata_lock_file is first_handle

    def test_other_lock_path_opens_and_closes(self, cache, tmp_path):
        lock_path = tmp_path / "other.lock"
        with cache._file_lock(lock_path):
            pass
        # Should not raise


class TestCacheCleanup:
    """Tests for __del__ cleanup."""

    def test_del_stops_worker(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        assert cache.worker_thread.is_alive()
        cache.__del__()
        time.sleep(1.5)
        assert not cache.worker_thread.is_alive()
