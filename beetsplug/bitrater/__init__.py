"""Beets plugin for detecting original MP3 bitrate using spectral analysis."""

# CRITICAL: Set thread limits BEFORE any imports that load numba/scipy/librosa
# These libraries check env vars at import time to initialize thread pools.
# This MUST happen before importing plugin (which imports analyzer -> spectrum -> librosa)
from beetsplug.bitrater._threading import clamp_threads

clamp_threads()

from beetsplug.bitrater.plugin import BitraterPlugin

__all__ = ['BitraterPlugin']
