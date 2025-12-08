# Transcode Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement hybrid transcode detection using SVM classifier + spectral cutoff validation to detect all files where stated quality exceeds actual quality.

**Architecture:** Two-stage detection: (1) SVM classifier predicts quality class from 150 PSD bands, (2) cutoff detector validates via sliding window band ratio comparison. Confidence penalties applied for disagreements. Transcode = stated quality rank > detected quality rank.

**Tech Stack:** Python 3.12, scipy (welch PSD), scikit-learn (SVM), numpy, pytest

---

## Task 1: Add V2 Class to Constants

**Files:**
- Modify: `beetsplug/bitrater/constants.py:14-25`
- Test: `tests/test_classifier.py`

**Step 1: Update BITRATE_CLASSES to include V2**

Edit `beetsplug/bitrater/constants.py`:

```python
# Classification classes (7 classes)
BITRATE_CLASSES = {
    0: ("128", 128),      # CBR 128 kbps - cutoff ~16 kHz
    1: ("V2", 190),       # VBR-2 (avg ~190 kbps) - cutoff ~18.5 kHz
    2: ("192", 192),      # CBR 192 kbps - cutoff ~19 kHz
    3: ("V0", 245),       # VBR-0 (avg ~245 kbps) - cutoff ~19.5 kHz
    4: ("256", 256),      # CBR 256 kbps - cutoff ~20 kHz
    5: ("320", 320),      # CBR 320 kbps - cutoff ~20.5 kHz
    6: ("LOSSLESS", 1411),  # Lossless (CD quality) - cutoff >21.5 kHz
}
```

**Step 2: Add quality ranking constant**

Add after CLASS_LABELS:

```python
# Quality ranking for transcode detection (higher = better quality)
QUALITY_RANK = {name: idx for idx, (name, _) in BITRATE_CLASSES.items()}
# Result: {"128": 0, "V2": 1, "192": 2, "V0": 3, "256": 4, "320": 5, "LOSSLESS": 6}

# Expected cutoff frequencies (Hz) for each class
CLASS_CUTOFFS = {
    "128": 16000,
    "V2": 18500,
    "192": 19000,
    "V0": 19500,
    "256": 20000,
    "320": 20500,
    "LOSSLESS": 22050,
}

# Cutoff detection tolerance (Hz)
CUTOFF_TOLERANCE = 500
```

**Step 3: Update training directory mapping in analyzer.py**

Edit `beetsplug/bitrater/analyzer.py` in `train_from_directory`:

```python
dir_to_class = {
    "128": CLASS_LABELS["128"],
    "v2": CLASS_LABELS["V2"],
    "192": CLASS_LABELS["192"],
    "v0": CLASS_LABELS["V0"],
    "256": CLASS_LABELS["256"],
    "320": CLASS_LABELS["320"],
    "lossless": CLASS_LABELS["LOSSLESS"],
}
```

**Step 4: Update test to expect 7 classes**

Edit `tests/test_classifier.py`:

```python
def test_seven_classes(self):
    """There should be exactly 7 quality classes."""
    assert len(BITRATE_CLASSES) == 7

def test_class_indices(self):
    """Class indices should be 0-6."""
    assert set(BITRATE_CLASSES.keys()) == {0, 1, 2, 3, 4, 5, 6}

def test_v2_class_exists(self):
    """V2 class should exist at index 1."""
    assert BITRATE_CLASSES[1] == ("V2", 190)
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add beetsplug/bitrater/constants.py beetsplug/bitrater/analyzer.py tests/test_classifier.py
git commit -m "feat: add V2 class and quality ranking constants"
```

---

## Task 2: Create CutoffDetector Class

**Files:**
- Create: `beetsplug/bitrater/cutoff_detector.py`
- Test: `tests/test_cutoff_detector.py`

**Step 1: Write failing test for cutoff detector initialization**

Create `tests/test_cutoff_detector.py`:

```python
"""Tests for cutoff detection."""

import numpy as np
import pytest

from beetsplug.bitrater.cutoff_detector import CutoffDetector, CutoffResult


class TestCutoffDetector:
    """Test CutoffDetector class."""

    def test_init(self):
        """CutoffDetector should initialize with default parameters."""
        detector = CutoffDetector()
        assert detector.min_freq == 15000
        assert detector.max_freq == 22050
        assert detector.coarse_step == 1000
        assert detector.fine_step == 100
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cutoff_detector.py::TestCutoffDetector::test_init -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'beetsplug.bitrater.cutoff_detector'"

**Step 3: Write minimal CutoffDetector class**

Create `beetsplug/bitrater/cutoff_detector.py`:

```python
"""Cutoff frequency detection for transcode validation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CutoffResult:
    """Result of cutoff frequency detection."""

    cutoff_frequency: int  # Detected cutoff in Hz
    gradient: float  # Sharpness of cutoff (higher = sharper = more artificial)
    is_sharp: bool  # True if gradient indicates artificial cutoff
    confidence: float  # Confidence in detection (0.0-1.0)


class CutoffDetector:
    """
    Detects frequency cutoff using sliding window band ratio comparison.

    Uses coarse-to-fine scanning to efficiently find where high-frequency
    content ends, then measures gradient sharpness to distinguish artificial
    MP3 cutoffs from natural rolloff.
    """

    def __init__(
        self,
        min_freq: int = 15000,
        max_freq: int = 22050,
        coarse_step: int = 1000,
        fine_step: int = 100,
        window_size: int = 1000,
        sharp_threshold: float = 0.5,
    ):
        """
        Initialize cutoff detector.

        Args:
            min_freq: Start of scan range (Hz)
            max_freq: End of scan range (Hz)
            coarse_step: Step size for initial scan (Hz)
            fine_step: Step size for refinement (Hz)
            window_size: Size of comparison windows (Hz)
            sharp_threshold: Gradient threshold for "sharp" classification
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.window_size = window_size
        self.sharp_threshold = sharp_threshold
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cutoff_detector.py::TestCutoffDetector::test_init -v`
Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/cutoff_detector.py tests/test_cutoff_detector.py
git commit -m "feat: add CutoffDetector class skeleton"
```

---

## Task 3: Implement Coarse Scan

**Files:**
- Modify: `beetsplug/bitrater/cutoff_detector.py`
- Test: `tests/test_cutoff_detector.py`

**Step 1: Write failing test for coarse scan**

Add to `tests/test_cutoff_detector.py`:

```python
def test_coarse_scan_finds_16khz_cutoff(self):
    """Coarse scan should find approximate cutoff at 16 kHz."""
    detector = CutoffDetector()

    # Create mock PSD: high energy below 16 kHz, noise floor above
    freqs = np.linspace(0, 22050, 2048)
    psd = np.ones_like(freqs)
    psd[freqs > 16000] = 0.001  # Sharp drop at 16 kHz

    candidate = detector._coarse_scan(psd, freqs)

    # Should find cutoff within 1 kHz of actual
    assert 15000 <= candidate <= 17000

def test_coarse_scan_finds_19khz_cutoff(self):
    """Coarse scan should find approximate cutoff at 19 kHz."""
    detector = CutoffDetector()

    freqs = np.linspace(0, 22050, 2048)
    psd = np.ones_like(freqs)
    psd[freqs > 19000] = 0.001

    candidate = detector._coarse_scan(psd, freqs)

    assert 18000 <= candidate <= 20000
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cutoff_detector.py -k "coarse_scan" -v`
Expected: FAIL with "AttributeError: 'CutoffDetector' object has no attribute '_coarse_scan'"

**Step 3: Implement coarse scan**

Add to `CutoffDetector` class in `beetsplug/bitrater/cutoff_detector.py`:

```python
def _coarse_scan(self, psd: np.ndarray, freqs: np.ndarray) -> int:
    """
    Perform coarse scan to find approximate cutoff region.

    Uses sliding window band ratio comparison at 1 kHz intervals.

    Args:
        psd: Power spectral density array
        freqs: Corresponding frequency array (Hz)

    Returns:
        Candidate cutoff frequency (Hz)
    """
    best_ratio = 1.0
    best_freq = self.max_freq

    for candidate_freq in range(self.min_freq, self.max_freq, self.coarse_step):
        # Window below candidate
        below_mask = (freqs >= candidate_freq - self.window_size) & (freqs < candidate_freq)
        # Window above candidate
        above_mask = (freqs >= candidate_freq) & (freqs < candidate_freq + self.window_size)

        if not np.any(below_mask) or not np.any(above_mask):
            continue

        energy_below = np.mean(psd[below_mask])
        energy_above = np.mean(psd[above_mask])

        # Avoid division by zero
        if energy_below < 1e-10:
            continue

        ratio = energy_above / energy_below

        # Find where ratio drops most dramatically
        if ratio < best_ratio:
            best_ratio = ratio
            best_freq = candidate_freq

    return best_freq
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cutoff_detector.py -k "coarse_scan" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/cutoff_detector.py tests/test_cutoff_detector.py
git commit -m "feat: implement coarse scan for cutoff detection"
```

---

## Task 4: Implement Fine Scan

**Files:**
- Modify: `beetsplug/bitrater/cutoff_detector.py`
- Test: `tests/test_cutoff_detector.py`

**Step 1: Write failing test for fine scan**

Add to `tests/test_cutoff_detector.py`:

```python
def test_fine_scan_refines_cutoff(self):
    """Fine scan should refine cutoff to within 100 Hz."""
    detector = CutoffDetector()

    # Create PSD with cutoff at exactly 16200 Hz
    freqs = np.linspace(0, 22050, 4096)
    psd = np.ones_like(freqs)
    psd[freqs > 16200] = 0.001

    # Start with coarse estimate
    coarse_estimate = 16000

    refined = detector._fine_scan(psd, freqs, coarse_estimate)

    # Should refine to within 200 Hz of actual cutoff
    assert 16000 <= refined <= 16400
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cutoff_detector.py::TestCutoffDetector::test_fine_scan_refines_cutoff -v`
Expected: FAIL

**Step 3: Implement fine scan**

Add to `CutoffDetector` class:

```python
def _fine_scan(self, psd: np.ndarray, freqs: np.ndarray, coarse_estimate: int) -> int:
    """
    Refine cutoff estimate with fine-grained scan.

    Scans ±500 Hz around coarse estimate at 100 Hz intervals.

    Args:
        psd: Power spectral density array
        freqs: Corresponding frequency array (Hz)
        coarse_estimate: Result from coarse scan (Hz)

    Returns:
        Refined cutoff frequency (Hz)
    """
    search_start = max(self.min_freq, coarse_estimate - 500)
    search_end = min(self.max_freq, coarse_estimate + 500)

    best_ratio = 1.0
    best_freq = coarse_estimate

    for candidate_freq in range(search_start, search_end, self.fine_step):
        below_mask = (freqs >= candidate_freq - self.window_size) & (freqs < candidate_freq)
        above_mask = (freqs >= candidate_freq) & (freqs < candidate_freq + self.window_size)

        if not np.any(below_mask) or not np.any(above_mask):
            continue

        energy_below = np.mean(psd[below_mask])
        energy_above = np.mean(psd[above_mask])

        if energy_below < 1e-10:
            continue

        ratio = energy_above / energy_below

        if ratio < best_ratio:
            best_ratio = ratio
            best_freq = candidate_freq

    return best_freq
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cutoff_detector.py::TestCutoffDetector::test_fine_scan_refines_cutoff -v`
Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/cutoff_detector.py tests/test_cutoff_detector.py
git commit -m "feat: implement fine scan for cutoff refinement"
```

---

## Task 5: Implement Gradient Measurement

**Files:**
- Modify: `beetsplug/bitrater/cutoff_detector.py`
- Test: `tests/test_cutoff_detector.py`

**Step 1: Write failing tests for gradient measurement**

Add to `tests/test_cutoff_detector.py`:

```python
def test_gradient_sharp_for_artificial_cutoff(self):
    """Sharp artificial cutoff should have high gradient."""
    detector = CutoffDetector()

    # Sharp step function at 16 kHz
    freqs = np.linspace(0, 22050, 4096)
    psd = np.ones_like(freqs)
    psd[freqs > 16000] = 0.001  # Instant drop

    gradient = detector._measure_gradient(psd, freqs, 16000)

    assert gradient > detector.sharp_threshold

def test_gradient_gradual_for_natural_rolloff(self):
    """Natural gradual rolloff should have low gradient."""
    detector = CutoffDetector()

    # Gradual rolloff - exponential decay starting at 14 kHz
    freqs = np.linspace(0, 22050, 4096)
    psd = np.ones_like(freqs)
    rolloff_mask = freqs > 14000
    psd[rolloff_mask] = np.exp(-0.0005 * (freqs[rolloff_mask] - 14000))

    gradient = detector._measure_gradient(psd, freqs, 16000)

    assert gradient < detector.sharp_threshold
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cutoff_detector.py -k "gradient" -v`
Expected: FAIL

**Step 3: Implement gradient measurement**

Add to `CutoffDetector` class:

```python
def _measure_gradient(self, psd: np.ndarray, freqs: np.ndarray, cutoff: int) -> float:
    """
    Measure gradient sharpness at the cutoff frequency.

    Calculates the slope of energy decline across the transition.
    Sharp gradients indicate artificial MP3 cutoffs.
    Gradual gradients suggest natural rolloff (old recordings, etc).

    Args:
        psd: Power spectral density array
        freqs: Corresponding frequency array (Hz)
        cutoff: Detected cutoff frequency (Hz)

    Returns:
        Gradient value (higher = sharper cutoff)
    """
    # Sample points 500 Hz below and above cutoff
    below_point = cutoff - 500
    above_point = cutoff + 500

    # Find energy at these points (average over small window)
    window = 200  # Hz

    below_mask = (freqs >= below_point - window/2) & (freqs < below_point + window/2)
    above_mask = (freqs >= above_point - window/2) & (freqs < above_point + window/2)

    if not np.any(below_mask) or not np.any(above_mask):
        return 0.0

    energy_below = np.mean(psd[below_mask])
    energy_above = np.mean(psd[above_mask])

    # Avoid log of zero
    if energy_below < 1e-10 or energy_above < 1e-10:
        if energy_below > energy_above:
            return 1.0  # Maximum sharpness
        return 0.0

    # Calculate gradient in dB per kHz
    db_drop = 10 * np.log10(energy_below / energy_above)
    gradient = db_drop / 1.0  # Per 1 kHz (1000 Hz span)

    # Normalize to 0-1 range (30 dB drop = 1.0)
    normalized = min(1.0, max(0.0, gradient / 30.0))

    return normalized
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cutoff_detector.py -k "gradient" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/cutoff_detector.py tests/test_cutoff_detector.py
git commit -m "feat: implement gradient sharpness measurement"
```

---

## Task 6: Implement Main detect() Method

**Files:**
- Modify: `beetsplug/bitrater/cutoff_detector.py`
- Test: `tests/test_cutoff_detector.py`

**Step 1: Write failing test for detect method**

Add to `tests/test_cutoff_detector.py`:

```python
def test_detect_128kbps_cutoff(self):
    """Detect should identify 128 kbps cutoff at ~16 kHz."""
    detector = CutoffDetector()

    freqs = np.linspace(0, 22050, 4096)
    psd = np.ones_like(freqs)
    psd[freqs > 16000] = 0.001

    result = detector.detect(psd, freqs)

    assert isinstance(result, CutoffResult)
    assert 15500 <= result.cutoff_frequency <= 16500
    assert result.is_sharp is True  # Artificial cutoff
    assert 0.0 <= result.confidence <= 1.0

def test_detect_lossless_no_cutoff(self):
    """Detect should identify lossless with cutoff > 21.5 kHz."""
    detector = CutoffDetector()

    freqs = np.linspace(0, 22050, 4096)
    psd = np.ones_like(freqs)  # Full spectrum, no cutoff

    result = detector.detect(psd, freqs)

    assert result.cutoff_frequency > 21000
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cutoff_detector.py -k "test_detect" -v`
Expected: FAIL

**Step 3: Implement detect method**

Add to `CutoffDetector` class:

```python
def detect(self, psd: np.ndarray, freqs: np.ndarray) -> CutoffResult:
    """
    Detect cutoff frequency using coarse-to-fine scanning.

    Args:
        psd: Power spectral density array
        freqs: Corresponding frequency array (Hz)

    Returns:
        CutoffResult with detected frequency and sharpness info
    """
    # Step 1: Coarse scan
    coarse_cutoff = self._coarse_scan(psd, freqs)

    # Step 2: Fine scan
    refined_cutoff = self._fine_scan(psd, freqs, coarse_cutoff)

    # Step 3: Measure gradient
    gradient = self._measure_gradient(psd, freqs, refined_cutoff)
    is_sharp = gradient > self.sharp_threshold

    # Step 4: Calculate confidence based on how clear the cutoff is
    # Higher gradient = clearer cutoff = higher confidence
    confidence = min(1.0, gradient + 0.3) if is_sharp else 0.5

    return CutoffResult(
        cutoff_frequency=refined_cutoff,
        gradient=gradient,
        is_sharp=is_sharp,
        confidence=confidence,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cutoff_detector.py -k "test_detect" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/cutoff_detector.py tests/test_cutoff_detector.py
git commit -m "feat: implement main detect() method for cutoff detection"
```

---

## Task 7: Update AnalysisResult Dataclass

**Files:**
- Modify: `beetsplug/bitrater/types.py`
- Test: `tests/test_analyzer.py`

**Step 1: Write failing test for new fields**

Add to `tests/test_analyzer.py`:

```python
class TestAnalysisResultFields:
    """Test new AnalysisResult fields for transcode detection."""

    def test_has_stated_class_field(self):
        """AnalysisResult should have stated_class field."""
        result = AnalysisResult(
            filename="test.flac",
            file_format="flac",
            original_format="192",
            original_bitrate=192,
            confidence=0.9,
            is_transcode=True,
            stated_class="LOSSLESS",
            detected_cutoff=19000,
            quality_gap=4,
        )
        assert result.stated_class == "LOSSLESS"

    def test_has_detected_cutoff_field(self):
        """AnalysisResult should have detected_cutoff field."""
        result = AnalysisResult(
            filename="test.mp3",
            file_format="mp3",
            original_format="320",
            original_bitrate=320,
            confidence=0.9,
            is_transcode=False,
            stated_class="320",
            detected_cutoff=20500,
            quality_gap=0,
        )
        assert result.detected_cutoff == 20500

    def test_has_quality_gap_field(self):
        """AnalysisResult should have quality_gap field."""
        result = AnalysisResult(
            filename="test.flac",
            file_format="flac",
            original_format="128",
            original_bitrate=128,
            confidence=0.85,
            is_transcode=True,
            stated_class="LOSSLESS",
            detected_cutoff=16000,
            quality_gap=6,  # LOSSLESS(6) - 128(0) = 6
        )
        assert result.quality_gap == 6
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_analyzer.py::TestAnalysisResultFields -v`
Expected: FAIL with TypeError about unexpected keyword arguments

**Step 3: Update AnalysisResult dataclass**

Edit `beetsplug/bitrater/types.py`:

```python
@dataclass
class AnalysisResult:
    """Result of analyzing an audio file."""

    # Core identification
    filename: str
    file_format: str  # Actual container: "mp3", "flac", "wav"

    # Classification results
    original_format: str  # "128", "V2", "192", "V0", "256", "320", "LOSSLESS"
    original_bitrate: int  # 128, 190, 192, 245, 256, 320, or 1411
    confidence: float  # Final confidence after penalties

    # Transcode detection
    is_transcode: bool  # True if stated_rank > detected_rank
    stated_class: str  # What file claims to be: "320", "LOSSLESS", etc.
    detected_cutoff: int  # Detected cutoff frequency in Hz
    quality_gap: int  # Difference in quality ranks (0-6)
    transcoded_from: Optional[str] = None  # e.g., "128" if transcoded

    # Metadata comparison
    stated_bitrate: Optional[int] = None  # What the file metadata claims

    # Analysis metadata
    analysis_version: str = "4.0"  # Updated version
    analysis_date: datetime = field(default_factory=datetime.now)
    warnings: List[str] = field(default_factory=list)

    def summarize(self) -> Dict[str, Any]:
        """Create a summary of key findings."""
        return {
            "filename": self.filename,
            "file_format": self.file_format,
            "original_format": self.original_format,
            "original_bitrate": self.original_bitrate,
            "confidence": self.confidence,
            "is_transcode": self.is_transcode,
            "stated_class": self.stated_class,
            "detected_cutoff": self.detected_cutoff,
            "quality_gap": self.quality_gap,
            "transcoded_from": self.transcoded_from,
            "stated_bitrate": self.stated_bitrate,
            "warnings": self.warnings,
            "analysis_date": self.analysis_date.isoformat(),
        }
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_analyzer.py::TestAnalysisResultFields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/types.py tests/test_analyzer.py
git commit -m "feat: add stated_class, detected_cutoff, quality_gap to AnalysisResult"
```

---

## Task 8: Implement Confidence Calculator

**Files:**
- Create: `beetsplug/bitrater/confidence.py`
- Test: `tests/test_confidence.py`

**Step 1: Write failing tests for confidence calculation**

Create `tests/test_confidence.py`:

```python
"""Tests for confidence calculation with penalties."""

import pytest

from beetsplug.bitrater.confidence import ConfidenceCalculator


class TestConfidenceCalculator:
    """Test confidence penalty calculations."""

    def test_no_penalty_when_cutoff_matches(self):
        """No penalty when detected cutoff matches expected."""
        calc = ConfidenceCalculator()

        # 320 kbps expects 20500 Hz, detected 20400 Hz (within tolerance)
        result = calc.calculate(
            classifier_confidence=0.9,
            detected_class="320",
            detected_cutoff=20400,
            gradient=0.8,  # Sharp
        )

        assert result.final_confidence >= 0.85  # Minimal penalty
        assert len(result.warnings) == 0

    def test_penalty_for_cutoff_mismatch(self):
        """Penalty applied when cutoff doesn't match class."""
        calc = ConfidenceCalculator()

        # Classifier says 320 (expects 20500 Hz), but cutoff is 19000 Hz (192 range)
        result = calc.calculate(
            classifier_confidence=0.9,
            detected_class="320",
            detected_cutoff=19000,
            gradient=0.8,
        )

        assert result.final_confidence < 0.7  # Significant penalty
        assert any("mismatch" in w.lower() for w in result.warnings)

    def test_penalty_for_soft_gradient(self):
        """Penalty applied for gradual rolloff (natural, not artificial)."""
        calc = ConfidenceCalculator()

        result = calc.calculate(
            classifier_confidence=0.9,
            detected_class="128",
            detected_cutoff=16000,
            gradient=0.2,  # Soft gradient
        )

        assert result.final_confidence < 0.85
        assert any("rolloff" in w.lower() or "natural" in w.lower() for w in result.warnings)

    def test_minimum_confidence_floor(self):
        """Confidence should never go below 0.1."""
        calc = ConfidenceCalculator()

        result = calc.calculate(
            classifier_confidence=0.3,  # Low to start
            detected_class="LOSSLESS",
            detected_cutoff=16000,  # Way off (should be >21500)
            gradient=0.2,  # Soft
        )

        assert result.final_confidence >= 0.1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_confidence.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Implement ConfidenceCalculator**

Create `beetsplug/bitrater/confidence.py`:

```python
"""Confidence calculation with penalty system."""

from dataclasses import dataclass, field
from typing import List

from .constants import CLASS_CUTOFFS, CUTOFF_TOLERANCE


@dataclass
class ConfidenceResult:
    """Result of confidence calculation."""

    final_confidence: float
    mismatch_penalty: float
    gradient_penalty: float
    warnings: List[str] = field(default_factory=list)


class ConfidenceCalculator:
    """
    Calculate final confidence using primary + penalties approach.

    Starts with classifier confidence, subtracts penalties for:
    - Cutoff mismatch (detected cutoff doesn't match expected for class)
    - Soft gradient (gradual rolloff suggests natural, not encoded)
    """

    def __init__(
        self,
        mismatch_penalty_small: float = 0.10,
        mismatch_penalty_medium: float = 0.25,
        mismatch_penalty_large: float = 0.40,
        gradient_penalty_moderate: float = 0.10,
        gradient_penalty_soft: float = 0.20,
        sharp_threshold: float = 0.5,
        minimum_confidence: float = 0.1,
    ):
        """Initialize with penalty values."""
        self.mismatch_penalty_small = mismatch_penalty_small
        self.mismatch_penalty_medium = mismatch_penalty_medium
        self.mismatch_penalty_large = mismatch_penalty_large
        self.gradient_penalty_moderate = gradient_penalty_moderate
        self.gradient_penalty_soft = gradient_penalty_soft
        self.sharp_threshold = sharp_threshold
        self.minimum_confidence = minimum_confidence

    def calculate(
        self,
        classifier_confidence: float,
        detected_class: str,
        detected_cutoff: int,
        gradient: float,
    ) -> ConfidenceResult:
        """
        Calculate final confidence with penalties.

        Args:
            classifier_confidence: Raw confidence from SVM classifier
            detected_class: Predicted class from classifier
            detected_cutoff: Cutoff frequency from cutoff detector
            gradient: Gradient sharpness from cutoff detector

        Returns:
            ConfidenceResult with final confidence and warnings
        """
        warnings: List[str] = []

        # Calculate mismatch penalty
        mismatch_penalty = self._calculate_mismatch_penalty(
            detected_class, detected_cutoff, warnings
        )

        # Calculate gradient penalty
        gradient_penalty = self._calculate_gradient_penalty(gradient, warnings)

        # Apply penalties
        final = classifier_confidence - mismatch_penalty - gradient_penalty
        final = max(self.minimum_confidence, final)

        return ConfidenceResult(
            final_confidence=final,
            mismatch_penalty=mismatch_penalty,
            gradient_penalty=gradient_penalty,
            warnings=warnings,
        )

    def _calculate_mismatch_penalty(
        self,
        detected_class: str,
        detected_cutoff: int,
        warnings: List[str],
    ) -> float:
        """Calculate penalty for cutoff mismatch."""
        if detected_class not in CLASS_CUTOFFS:
            return 0.0

        expected_cutoff = CLASS_CUTOFFS[detected_class]
        difference = abs(detected_cutoff - expected_cutoff)

        if difference <= CUTOFF_TOLERANCE:
            return 0.0
        elif difference <= CUTOFF_TOLERANCE * 2:
            return self.mismatch_penalty_small
        elif difference <= CUTOFF_TOLERANCE * 4:
            warnings.append(
                f"Cutoff mismatch: expected ~{expected_cutoff} Hz, "
                f"found {detected_cutoff} Hz"
            )
            return self.mismatch_penalty_medium
        else:
            warnings.append(
                f"Significant cutoff mismatch: expected ~{expected_cutoff} Hz, "
                f"found {detected_cutoff} Hz"
            )
            return self.mismatch_penalty_large

    def _calculate_gradient_penalty(
        self,
        gradient: float,
        warnings: List[str],
    ) -> float:
        """Calculate penalty for soft gradient (natural rolloff)."""
        if gradient >= self.sharp_threshold:
            return 0.0
        elif gradient >= self.sharp_threshold * 0.5:
            return self.gradient_penalty_moderate
        else:
            warnings.append(
                "Gradual frequency rolloff detected - may be natural, not encoded"
            )
            return self.gradient_penalty_soft
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_confidence.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/confidence.py tests/test_confidence.py
git commit -m "feat: implement confidence calculator with penalty system"
```

---

## Task 9: Implement Quality Ranking and Transcode Logic

**Files:**
- Create: `beetsplug/bitrater/transcode_detector.py`
- Test: `tests/test_transcode_detector.py`

**Step 1: Write failing tests for transcode detection**

Create `tests/test_transcode_detector.py`:

```python
"""Tests for transcode detection logic."""

import pytest

from beetsplug.bitrater.transcode_detector import TranscodeDetector


class TestTranscodeDetector:
    """Test transcode detection based on quality ranking."""

    def test_flac_from_128_is_transcode(self):
        """FLAC with 128 kbps content should be detected as transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="LOSSLESS",
            detected_class="128",
        )

        assert result.is_transcode is True
        assert result.quality_gap == 6  # LOSSLESS(6) - 128(0)
        assert result.transcoded_from == "128"

    def test_mp3_320_from_192_is_transcode(self):
        """320 kbps MP3 with 192 kbps content is transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="192",
        )

        assert result.is_transcode is True
        assert result.quality_gap == 3  # 320(5) - 192(2)
        assert result.transcoded_from == "192"

    def test_genuine_320_is_not_transcode(self):
        """Genuine 320 kbps file is not a transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="320",
        )

        assert result.is_transcode is False
        assert result.quality_gap == 0
        assert result.transcoded_from is None

    def test_192_detected_as_320_is_not_transcode(self):
        """File claiming lower quality than detected is not transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="192",
            detected_class="320",
        )

        assert result.is_transcode is False
        assert result.quality_gap == 0  # No gap when detected > stated
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcode_detector.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Implement TranscodeDetector**

Create `beetsplug/bitrater/transcode_detector.py`:

```python
"""Transcode detection based on quality ranking."""

from dataclasses import dataclass
from typing import Optional

from .constants import QUALITY_RANK


@dataclass
class TranscodeResult:
    """Result of transcode detection."""

    is_transcode: bool
    quality_gap: int  # 0-6, higher = more severe
    transcoded_from: Optional[str]  # Original quality if transcode


class TranscodeDetector:
    """
    Detect transcodes by comparing stated vs detected quality.

    Transcode = stated quality rank > detected quality rank
    """

    def detect(
        self,
        stated_class: str,
        detected_class: str,
    ) -> TranscodeResult:
        """
        Determine if file is a transcode.

        Args:
            stated_class: What the file claims to be (from container/metadata)
            detected_class: What the classifier detected

        Returns:
            TranscodeResult with detection info
        """
        stated_rank = QUALITY_RANK.get(stated_class, 0)
        detected_rank = QUALITY_RANK.get(detected_class, 0)

        is_transcode = stated_rank > detected_rank
        quality_gap = stated_rank - detected_rank if is_transcode else 0
        transcoded_from = detected_class if is_transcode else None

        return TranscodeResult(
            is_transcode=is_transcode,
            quality_gap=quality_gap,
            transcoded_from=transcoded_from,
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transcode_detector.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/transcode_detector.py tests/test_transcode_detector.py
git commit -m "feat: implement transcode detection with quality ranking"
```

---

## Task 10: Integrate Components into AudioQualityAnalyzer

**Files:**
- Modify: `beetsplug/bitrater/analyzer.py`
- Test: `tests/test_analyzer.py`

**Step 1: Write failing integration test**

Add to `tests/test_analyzer.py`:

```python
class TestIntegratedTranscodeDetection:
    """Test full transcode detection pipeline."""

    def test_analyze_detects_stated_class_from_container(self):
        """Analyzer should determine stated_class from file format."""
        # This test will need mock spectral data
        # For now, test the helper method
        from beetsplug.bitrater.analyzer import AudioQualityAnalyzer

        analyzer = AudioQualityAnalyzer()

        # FLAC container = LOSSLESS stated class
        assert analyzer._get_stated_class("flac", None) == "LOSSLESS"
        # WAV container = LOSSLESS stated class
        assert analyzer._get_stated_class("wav", None) == "LOSSLESS"
        # MP3 with 320 bitrate
        assert analyzer._get_stated_class("mp3", 320) == "320"
        # MP3 with 192 bitrate
        assert analyzer._get_stated_class("mp3", 192) == "192"
        # MP3 with ~245 bitrate (V0 range)
        assert analyzer._get_stated_class("mp3", 245) == "V0"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_analyzer.py::TestIntegratedTranscodeDetection -v`
Expected: FAIL with AttributeError

**Step 3: Add _get_stated_class helper method**

Add to `AudioQualityAnalyzer` class in `beetsplug/bitrater/analyzer.py`:

```python
def _get_stated_class(self, file_format: str, stated_bitrate: Optional[int]) -> str:
    """
    Determine stated quality class from file format and metadata.

    Args:
        file_format: Container format (mp3, flac, etc.)
        stated_bitrate: Bitrate from file metadata (if available)

    Returns:
        Quality class string: "128", "V2", "192", "V0", "256", "320", "LOSSLESS"
    """
    # Lossless containers are always stated as LOSSLESS
    if file_format in LOSSLESS_CONTAINERS:
        return "LOSSLESS"

    # For lossy containers, use bitrate to determine stated class
    if stated_bitrate is None:
        return "UNKNOWN"

    # Map bitrate to class
    if stated_bitrate <= 140:
        return "128"
    elif stated_bitrate <= 175:
        return "V2"
    elif stated_bitrate <= 210:
        return "192"
    elif stated_bitrate <= 260:
        return "V0"
    elif stated_bitrate <= 290:
        return "256"
    else:
        return "320"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_analyzer.py::TestIntegratedTranscodeDetection -v`
Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/analyzer.py tests/test_analyzer.py
git commit -m "feat: add _get_stated_class helper for determining stated quality"
```

---

## Task 11: Update analyze_file to Use New Components

**Files:**
- Modify: `beetsplug/bitrater/analyzer.py`
- Modify: `beetsplug/bitrater/spectrum.py` (add get_psd method)
- Test: `tests/test_analyzer.py`

**Step 1: Add get_psd method to SpectrumAnalyzer**

Add to `SpectrumAnalyzer` class in `beetsplug/bitrater/spectrum.py`:

```python
def get_psd(self, file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Get raw PSD data for cutoff detection.

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (psd, freqs) arrays, or None if analysis fails
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return None

    if not self._validate_audio(y, sr):
        return None

    # Calculate PSD
    freqs, psd = signal.welch(y, sr, nperseg=self.fft_size)

    return psd, freqs
```

**Step 2: Update analyze_file method**

Replace the `analyze_file` method in `AudioQualityAnalyzer`:

```python
def analyze_file(self, file_path: str) -> Optional[AnalysisResult]:
    """
    Analyze a single audio file with hybrid transcode detection.

    Performs spectral analysis, SVM classification, cutoff detection,
    and applies confidence penalties.
    """
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    # 1. Get file metadata
    try:
        metadata = self.file_analyzer.analyze(file_path)
    except Exception as e:
        logger.warning(f"Could not read metadata from {file_path}: {e}")
        metadata = None

    # 2. Extract spectral features
    features = self.spectrum_analyzer.analyze_file(file_path)
    if features is None:
        logger.error(f"Failed to extract spectral features from {file_path}")
        return None

    # 3. Get file format and stated class
    file_format = path.suffix.lower().lstrip(".")
    stated_bitrate = metadata.bitrate if metadata else None
    stated_class = self._get_stated_class(file_format, stated_bitrate)

    # 4. Classify (if model is trained)
    if not self.classifier.trained:
        logger.warning("Classifier not trained - returning features without classification")
        return AnalysisResult(
            filename=str(path),
            file_format=file_format,
            original_format="UNKNOWN",
            original_bitrate=0,
            confidence=0.0,
            is_transcode=False,
            stated_class=stated_class,
            detected_cutoff=0,
            quality_gap=0,
            stated_bitrate=stated_bitrate,
            warnings=["Classifier not trained"],
        )

    prediction = self.classifier.predict(features)

    # 5. Cutoff detection for validation
    psd_data = self.spectrum_analyzer.get_psd(file_path)
    if psd_data is not None:
        psd, freqs = psd_data
        cutoff_result = self.cutoff_detector.detect(psd, freqs)
        detected_cutoff = cutoff_result.cutoff_frequency
        gradient = cutoff_result.gradient
    else:
        detected_cutoff = 0
        gradient = 0.5  # Neutral

    # 6. Calculate confidence with penalties
    conf_result = self.confidence_calculator.calculate(
        classifier_confidence=prediction.confidence,
        detected_class=prediction.format_type,
        detected_cutoff=detected_cutoff,
        gradient=gradient,
    )

    # 7. Detect transcode
    transcode_result = self.transcode_detector.detect(
        stated_class=stated_class,
        detected_class=prediction.format_type,
    )

    # 8. Collect all warnings
    warnings = list(conf_result.warnings)
    if transcode_result.is_transcode:
        warnings.append(
            f"File appears to be transcoded from {transcode_result.transcoded_from} "
            f"(quality gap: {transcode_result.quality_gap})"
        )

    return AnalysisResult(
        filename=str(path),
        file_format=file_format,
        original_format=prediction.format_type,
        original_bitrate=prediction.estimated_bitrate,
        confidence=conf_result.final_confidence,
        is_transcode=transcode_result.is_transcode,
        stated_class=stated_class,
        detected_cutoff=detected_cutoff,
        quality_gap=transcode_result.quality_gap,
        transcoded_from=transcode_result.transcoded_from,
        stated_bitrate=stated_bitrate,
        warnings=warnings,
    )
```

**Step 3: Update __init__ to create new components**

Update `AudioQualityAnalyzer.__init__`:

```python
def __init__(self, model_path: Optional[Path] = None):
    """Initialize analyzer components."""
    self.spectrum_analyzer = SpectrumAnalyzer()
    self.classifier = QualityClassifier(model_path)
    self.file_analyzer = FileAnalyzer()
    self.cutoff_detector = CutoffDetector()
    self.confidence_calculator = ConfidenceCalculator()
    self.transcode_detector = TranscodeDetector()
```

**Step 4: Add imports at top of analyzer.py**

```python
from .cutoff_detector import CutoffDetector
from .confidence import ConfidenceCalculator
from .transcode_detector import TranscodeDetector
```

**Step 5: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add beetsplug/bitrater/analyzer.py beetsplug/bitrater/spectrum.py
git commit -m "feat: integrate cutoff detector and confidence system into analyzer"
```

---

## Task 12: Fix Existing Tests for New API

**Files:**
- Modify: `tests/test_analyzer.py`
- Modify: `tests/test_classifier.py`

**Step 1: Update test fixtures for new AnalysisResult fields**

Update any existing tests that create `AnalysisResult` to include new required fields.

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Fix any failures by updating tests to match new API.

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: update existing tests for new transcode detection API"
```

---

## Task 13: Final Integration Test

**Files:**
- Test: `tests/test_integration.py`

**Step 1: Write end-to-end integration test**

Create `tests/test_integration.py`:

```python
"""End-to-end integration tests for transcode detection."""

import numpy as np
import pytest

from beetsplug.bitrater.cutoff_detector import CutoffDetector
from beetsplug.bitrater.confidence import ConfidenceCalculator
from beetsplug.bitrater.transcode_detector import TranscodeDetector
from beetsplug.bitrater.constants import QUALITY_RANK, CLASS_CUTOFFS


class TestFullPipeline:
    """Test complete transcode detection pipeline."""

    def test_128_to_flac_detection_pipeline(self):
        """Full pipeline should detect 128 kbps source in FLAC container."""
        # Simulate 128 kbps spectral signature
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)
        psd[freqs > 16000] = 0.001  # 128 kbps cutoff

        # Step 1: Cutoff detection
        cutoff_detector = CutoffDetector()
        cutoff_result = cutoff_detector.detect(psd, freqs)

        assert 15500 <= cutoff_result.cutoff_frequency <= 16500
        assert cutoff_result.is_sharp is True

        # Step 2: Confidence calculation (simulating classifier said "128")
        conf_calc = ConfidenceCalculator()
        conf_result = conf_calc.calculate(
            classifier_confidence=0.9,
            detected_class="128",
            detected_cutoff=cutoff_result.cutoff_frequency,
            gradient=cutoff_result.gradient,
        )

        # Cutoff matches, so no major penalty
        assert conf_result.final_confidence >= 0.8

        # Step 3: Transcode detection (FLAC stated, 128 detected)
        transcode_detector = TranscodeDetector()
        transcode_result = transcode_detector.detect(
            stated_class="LOSSLESS",
            detected_class="128",
        )

        assert transcode_result.is_transcode is True
        assert transcode_result.quality_gap == 6
        assert transcode_result.transcoded_from == "128"

    def test_genuine_lossless_pipeline(self):
        """Full pipeline should correctly identify genuine lossless."""
        # Simulate lossless spectral signature (full spectrum)
        freqs = np.linspace(0, 22050, 4096)
        psd = np.ones_like(freqs)  # Full spectrum, no artificial cutoff

        cutoff_detector = CutoffDetector()
        cutoff_result = cutoff_detector.detect(psd, freqs)

        # Should find cutoff at or above 21 kHz
        assert cutoff_result.cutoff_frequency > 20000

        # Transcode detection (FLAC stated, LOSSLESS detected)
        transcode_detector = TranscodeDetector()
        transcode_result = transcode_detector.detect(
            stated_class="LOSSLESS",
            detected_class="LOSSLESS",
        )

        assert transcode_result.is_transcode is False
        assert transcode_result.quality_gap == 0
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for transcode detection"
```

---

## Summary

After completing all tasks, the transcode detection system will:

1. **Support 7 quality classes**: 128, V2, 192, V0, 256, 320, LOSSLESS
2. **Detect cutoffs** using sliding window band ratio comparison with coarse-to-fine scanning
3. **Measure gradient sharpness** to distinguish artificial vs natural rolloff
4. **Calculate confidence** with penalties for cutoff mismatch and soft gradients
5. **Detect all transcodes** where stated quality exceeds detected quality (not just lossless containers)
6. **Report quality gap** for severity-based sorting in beets

**Files created:**
- `beetsplug/bitrater/cutoff_detector.py`
- `beetsplug/bitrater/confidence.py`
- `beetsplug/bitrater/transcode_detector.py`
- `tests/test_cutoff_detector.py`
- `tests/test_confidence.py`
- `tests/test_transcode_detector.py`
- `tests/test_integration.py`

**Files modified:**
- `beetsplug/bitrater/constants.py`
- `beetsplug/bitrater/types.py`
- `beetsplug/bitrater/spectrum.py`
- `beetsplug/bitrater/analyzer.py`
- `tests/test_analyzer.py`
- `tests/test_classifier.py`
