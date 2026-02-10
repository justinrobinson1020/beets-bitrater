"""Beets plugin for bitrater — wraps the standalone bitrater library."""

from bitrater._threading import clamp_threads

clamp_threads()

from beetsplug.bitrater.plugin import BitraterPlugin

__all__ = ["BitraterPlugin"]
