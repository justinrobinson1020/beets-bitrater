"""Thread-safe cache system with improved concurrency handling."""

import fcntl
import hashlib
import json
import logging
import queue
import threading
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FeatureCache:
    """Thread-safe manager for caching of extracted spectral features."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.features_dir = self.cache_dir / "features"
        self.metadata_path = self.cache_dir / "cache_metadata.json"
        self.metadata: dict[str, Any] = {}

        # Use RLock instead of Lock to allow recursive locking
        self.metadata_lock = RLock()

        # Create update queue and worker thread
        self.update_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._metadata_worker, daemon=True)
        self.worker_running = True

        # Cached lock file handle for metadata (avoids open/close per operation)
        self._metadata_lock_path = self.metadata_path.with_suffix(".lock")
        self._metadata_lock_file: Any = None

        # Create cache directories
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata
        self._load_metadata()

        # Start worker thread
        self.worker_thread.start()

    def _get_file_key(self, file_path: Path) -> str:
        """Generate cache key from path, mtime, and size (no file read)."""
        stat = file_path.stat()
        key_data = f"{file_path.resolve()}|{stat.st_mtime}|{stat.st_size}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _get_cache_path(self, file_hash: str) -> Path:
        """Get path for cached features file."""
        return self.features_dir / f"{file_hash}.npz"

    @contextmanager
    def _file_lock(self, lock_path: Path):
        """Cross-process file lock using fcntl with cached file handle."""
        # Use cached handle for metadata lock path (most common case)
        if lock_path == self._metadata_lock_path:
            if self._metadata_lock_file is None:
                self._metadata_lock_file = open(lock_path, "w")
            lock_file = self._metadata_lock_file
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        else:
            # Fallback for other lock paths
            lock_file = open(lock_path, "w")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()

    def _metadata_worker(self):
        """Worker thread to handle metadata updates."""
        while self.worker_running:
            try:
                # Get update from queue with timeout
                try:
                    update_func = self.update_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Execute update with timeout
                try:
                    with self.metadata_lock:
                        update_func()
                    self._save_metadata()
                finally:
                    self.update_queue.task_done()

            except Exception as e:
                logger.error(f"Error in metadata worker: {e}")

    def _load_metadata(self) -> None:
        """Load cache metadata from disk with cross-process locking."""
        lock_path = self.metadata_path.with_suffix(".lock")
        try:
            if self.metadata_path.exists():
                with self._file_lock(lock_path):
                    with open(self.metadata_path) as f:
                        with self.metadata_lock:
                            self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
            self.metadata = {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk with cross-process locking."""
        lock_path = self.metadata_path.with_suffix(".lock")
        try:
            # Deepcopy OUTSIDE lock to minimize critical section time
            with self.metadata_lock:
                metadata_copy = deepcopy(self.metadata)

            with self._file_lock(lock_path):
                temp_path = self.metadata_path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(metadata_copy, f, indent=2)
                temp_path.replace(self.metadata_path)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")

    def get_features(self, file_path: Path) -> tuple[np.ndarray, dict] | None:
        """Get cached features with timeout."""
        try:
            file_hash = self._get_file_key(file_path)
            cache_path = self._get_cache_path(file_hash)

            if not cache_path.exists():
                return None

            # Check modification time with timeout
            try:
                with self.metadata_lock:
                    metadata_entry = self.metadata.get(file_hash, {})
                    cache_mtime = metadata_entry.get("cached_date")
            except Exception as e:
                logger.error(f"Error checking cache time for {file_path}: {e}")
                return None

            if cache_mtime:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                cache_mtime = datetime.fromisoformat(cache_mtime)
                if file_mtime > cache_mtime:
                    return None

            # Load features (no pickle required)
            with np.load(cache_path, allow_pickle=False) as data:
                features = data["features"]
                if "metadata_json" in data:
                    metadata = json.loads(data["metadata_json"].tobytes().decode("utf-8"))
                else:
                    # Legacy format — force re-cache
                    return None

            return features, metadata

        except Exception as e:
            logger.error(f"Error retrieving cached features for {file_path}: {e}")
            return None

    def save_features(self, file_path: Path, features: np.ndarray, metadata: dict) -> None:
        """Save features with queued metadata update."""
        try:
            file_hash = self._get_file_key(file_path)
            cache_path = self._get_cache_path(file_hash)

            # Encode metadata as JSON bytes (avoids pickle entirely)
            metadata_bytes = json.dumps(metadata, default=str).encode("utf-8")
            metadata_arr = np.frombuffer(metadata_bytes, dtype=np.uint8).copy()
            np.savez(
                cache_path,
                features=features,
                metadata_json=metadata_arr,
            )

            # Queue metadata update
            def update_metadata():
                self.metadata[file_hash] = {
                    "file_path": str(file_path),
                    "cached_date": datetime.now().isoformat(),
                    "feature_shape": features.shape,
                }

            self.update_queue.put(update_metadata)

        except Exception as e:
            logger.error(f"Error caching features for {file_path}: {e}")

    def clear_cache(self) -> None:
        """Clear cache with queued update."""
        try:
            # Clear files
            for cache_file in self.features_dir.glob("*.npz"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Error deleting cache file: {e}")

            # Queue metadata clear
            def clear_metadata():
                self.metadata = {}

            self.update_queue.put(clear_metadata)
            logger.info("Feature cache cleared")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache info with timeout."""
        try:
            with self.metadata_lock:
                metadata_values = list(self.metadata.values())
                total_size = sum(f.stat().st_size for f in self.features_dir.glob("*.npz"))

                last_modified = (
                    max(
                        (datetime.fromisoformat(m["cached_date"]) for m in metadata_values),
                        default=None,
                    )
                    if metadata_values
                    else None
                )

                return {
                    "total_files": len(self.metadata),
                    "total_size_mb": total_size / (1024 * 1024),
                    "cache_dir": str(self.cache_dir),
                    "last_modified": last_modified,
                }
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {}

    def __del__(self):
        """Clean shutdown of worker thread and cached file handles."""
        self.worker_running = False
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        # Close cached lock file handle
        if hasattr(self, "_metadata_lock_file") and self._metadata_lock_file is not None:
            try:
                self._metadata_lock_file.close()
            except Exception:
                pass
