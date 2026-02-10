"""Standalone audio quality analysis library."""

from bitrater._threading import clamp_threads

clamp_threads()

from bitrater.analyzer import AudioQualityAnalyzer
from bitrater.types import AnalysisResult, SpectralFeatures

__all__ = ["AudioQualityAnalyzer", "AnalysisResult", "SpectralFeatures"]
