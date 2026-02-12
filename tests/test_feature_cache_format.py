"""Tests for feature cache format changes (no allow_pickle)."""

import json

import numpy as np

from bitrater.feature_cache import FeatureCache


class TestFeatureCacheNoPickle:
    """Tests that cache uses JSON metadata instead of pickled objects."""

    def test_save_and_get_roundtrip(self, tmp_path) -> None:
        """Features and metadata should survive a save/get roundtrip."""
        cache = FeatureCache(tmp_path / "cache")
        features = np.random.rand(193).astype(np.float32)
        metadata = {
            "n_bands": 150,
            "approach": "encoder_agnostic_v12",
            "sample_rate": 44100,
            "creation_date": "2026-01-01T00:00:00",
            "cutoff_len": 6,
            "sfb21_len": 6,
            "rolloff_len": 4,
            "discriminative_len": 8,
            "band_frequencies": [(16000.0, 16040.0)] * 150,
        }

        # Create a real audio file path for cache key
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        cache.save_features(audio_file, features, metadata)
        # Wait for async worker to process
        cache.update_queue.join()

        result = cache.get_features(audio_file)
        assert result is not None

        loaded_features, loaded_metadata = result
        np.testing.assert_array_almost_equal(loaded_features, features, decimal=5)
        assert loaded_metadata["n_bands"] == 150
        assert loaded_metadata["approach"] == "encoder_agnostic_v12"

    def test_saved_npz_loads_without_allow_pickle(self, tmp_path) -> None:
        """Saved cache file should be loadable with allow_pickle=False."""
        cache = FeatureCache(tmp_path / "cache")
        features = np.random.rand(193).astype(np.float32)
        metadata = {"n_bands": 150, "approach": "encoder_agnostic_v12"}

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        cache.save_features(audio_file, features, metadata)
        cache.update_queue.join()

        # Find the cached .npz file
        npz_files = list((tmp_path / "cache" / "features").glob("*.npz"))
        assert len(npz_files) == 1

        # Loading with allow_pickle=False should succeed
        with np.load(npz_files[0], allow_pickle=False) as data:
            assert "features" in data
            assert "metadata_json" in data

    def test_metadata_stored_as_json_bytes(self, tmp_path) -> None:
        """Metadata should be encoded as JSON bytes, not pickled objects."""
        cache = FeatureCache(tmp_path / "cache")
        features = np.random.rand(193).astype(np.float32)
        metadata = {"n_bands": 150, "approach": "encoder_agnostic_v12"}

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        cache.save_features(audio_file, features, metadata)
        cache.update_queue.join()

        # Find and load the cached .npz file
        npz_files = list((tmp_path / "cache" / "features").glob("*.npz"))
        with np.load(npz_files[0], allow_pickle=False) as data:
            # Decode metadata_json back to dict
            json_bytes = data["metadata_json"].tobytes()
            loaded = json.loads(json_bytes.decode("utf-8"))
            assert loaded["n_bands"] == 150
            assert loaded["approach"] == "encoder_agnostic_v12"

    def test_legacy_format_returns_none(self, tmp_path) -> None:
        """Old pickled format should return None (force re-cache)."""
        cache = FeatureCache(tmp_path / "cache")

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        # Manually create a legacy-format cache file
        file_hash = cache._get_file_key(audio_file)
        cache_path = cache._get_cache_path(file_hash)

        features = np.random.rand(193).astype(np.float32)
        metadata = {"n_bands": 150}
        metadata_arr = np.array(metadata, dtype=object)
        np.savez(cache_path, features=features, metadata=metadata_arr)

        # Attempting to load should return None (not crash)
        result = cache.get_features(audio_file)
        assert result is None
