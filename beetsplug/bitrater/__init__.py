"""Beets plugin for detecting original MP3 bitrate using spectral analysis."""

# CRITICAL: Set thread limits BEFORE any imports that load numba/scipy/librosa
# These libraries check env vars at import time to initialize thread pools.
# This MUST happen before importing plugin (which imports analyzer -> spectrum -> librosa)
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from beetsplug.bitrater.plugin import BitraterPlugin

__all__ = ['BitraterPlugin']
