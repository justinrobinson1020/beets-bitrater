"""Standalone audio quality analysis library."""

from bitrater._threading import clamp_threads

clamp_threads()

from bitrater.analyzer import AudioQualityAnalyzer  # noqa: E402
from bitrater.types import AnalysisResult, SpectralFeatures  # noqa: E402

__all__ = ["AudioQualityAnalyzer", "AnalysisResult", "SpectralFeatures"]
