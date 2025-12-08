# Transcode Detection Design

## Overview

This document describes the hybrid transcode detection system for the beets-bitrater plugin. The system detects files where stated quality exceeds actual quality, including:

- Lossless files (FLAC/WAV) created from lossy sources
- High-bitrate MP3s upconverted from lower-bitrate sources
- Any file claiming higher quality than its spectral content indicates

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│  Audio file (any format: FLAC, WAV, MP3, AAC, etc.)             │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ File Metadata   │  │ SVM Classifier  │  │ Cutoff Detector │
│                 │  │                 │  │                 │
│ • Container     │  │ • 7 classes     │  │ • Band ratios   │
│ • Stated bitrate│  │ • 150 PSD bands │  │ • Sliding window│
│ • Stated class  │  │ • Poly kernel   │  │ • Coarse→fine   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIDENCE CALCULATOR                         │
│                                                                  │
│  final_conf = classifier_conf - mismatch_penalty - gradient_pen │
│                                                                  │
│  is_transcode = QUALITY_RANK[stated] > QUALITY_RANK[detected]   │
│  quality_gap = stated_rank - detected_rank                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                    │
│                                                                  │
│  • detected_class: "128" | "V2" | "192" | "V0" | "256" | "320"  │
│  • detected_cutoff: frequency in Hz                             │
│  • is_transcode: True/False                                     │
│  • quality_gap: 0-6 (severity)                                  │
│  • confidence: 0.0-1.0                                          │
│  • warnings: list of discrepancy messages                       │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. SVM Classifier (Primary)

The classifier uses the D'Alessandro & Shi methodology with extensions for lossless detection.

**Classes (7 total):**

| Class | Cutoff | Avg Bitrate |
|-------|--------|-------------|
| 128 | 16 kHz | 128 kbps |
| V2 | 18.5 kHz | ~190 kbps |
| 192 | 19 kHz | 192 kbps |
| V0 | 19.5 kHz | ~245 kbps |
| 256 | 20 kHz | 256 kbps |
| 320 | 20.5 kHz | 320 kbps |
| LOSSLESS | 22+ kHz | ~1411 kbps |

**Features:**
- 150 PSD bands covering 16-22 kHz
- Bands 0-99: Paper's bitrate detection range (16-20 kHz)
- Bands 100-149: Ultrasonic extension for lossless detection (20-22 kHz)

**Model:**
- SVM with polynomial kernel (degree=2, γ=1, C=1)
- StandardScaler for feature normalization
- Probability output for confidence scores

### 2. Cutoff Detector (Validation)

Independently detects the frequency cutoff to validate classifier output.

**Method: Sliding Window Band Ratio Comparison**

```
Step 1: Coarse Scan (1 kHz intervals from 15-22 kHz)
────────────────────────────────────────────────────
For each candidate frequency F:
  - Window BELOW: average energy from (F - 1kHz) to F
  - Window ABOVE: average energy from F to (F + 1kHz)
  - Ratio = ABOVE / BELOW

Find F where ratio drops most dramatically.

Step 2: Fine Scan (100 Hz intervals around candidate)
────────────────────────────────────────────────────
Scan ±500 Hz around the coarse candidate.
Pinpoint exact cutoff frequency.

Step 3: Gradient Sharpness Measurement
────────────────────────────────────────────────────
Calculate slope of energy decline across the transition:
  - Sample energy at cutoff - 500 Hz
  - Sample energy at cutoff + 500 Hz
  - Gradient = (energy_below - energy_above) / 1000 Hz

Sharp gradient (> threshold) = artificial MP3 cutoff
Gradual gradient = natural rolloff (old recording, etc.)
```

**Expected Cutoff Ranges:**

| Class | Expected Cutoff | Tolerance |
|-------|-----------------|-----------|
| 128 | 16.0 kHz | ±0.5 kHz |
| V2 | 18.5 kHz | ±0.5 kHz |
| 192 | 19.0 kHz | ±0.5 kHz |
| V0 | 19.5 kHz | ±0.5 kHz |
| 256 | 20.0 kHz | ±0.5 kHz |
| 320 | 20.5 kHz | ±0.5 kHz |
| LOSSLESS | >21.5 kHz | N/A |

### 3. Confidence Calculator

**Formula:**
```
final_confidence = classifier_confidence
                   - cutoff_mismatch_penalty
                   - soft_gradient_penalty
```

**Penalty 1: Cutoff Mismatch**

| Scenario | Penalty |
|----------|---------|
| Cutoff within ±0.5 kHz of expected | 0.0 |
| Cutoff within ±1.0 kHz of expected | 0.10 |
| Cutoff off by >1.0 kHz | 0.25 |
| Cutoff suggests different class entirely | 0.40 |

**Penalty 2: Soft Gradient (Natural Rolloff)**

| Gradient Sharpness | Interpretation | Penalty |
|--------------------|----------------|---------|
| Very sharp (steep slope) | Artificial cutoff | 0.0 |
| Moderate | Ambiguous | 0.10 |
| Gradual (shallow slope) | Natural rolloff | 0.20 |

**Constraints:**
- Minimum confidence floor: 0.1
- Warning generation: Any penalty >0.15 triggers user-visible warning

## Transcode Detection Logic

**Core Principle:** Transcode = stated quality > detected quality

```python
QUALITY_RANK = {
    "128": 0,
    "V2": 1,
    "192": 2,
    "V0": 3,
    "256": 4,
    "320": 5,
    "LOSSLESS": 6,
}

def detect_transcode(file_path):
    # 1. Get stated quality from file
    container = get_container_format(file_path)
    stated_bitrate = get_stated_bitrate(file_path)
    stated_class = determine_stated_class(container, stated_bitrate)

    # 2. Run classifier → detected quality
    detected_class = classifier.predict(features)

    # 3. Run cutoff detector for validation
    detected_cutoff = cutoff_detector.find_cutoff(features)
    gradient = cutoff_detector.measure_gradient()

    # 4. Transcode detection
    stated_rank = QUALITY_RANK[stated_class]
    detected_rank = QUALITY_RANK[detected_class]

    is_transcode = stated_rank > detected_rank
    quality_gap = stated_rank - detected_rank if is_transcode else 0

    # 5. Calculate confidence with penalties
    expected_cutoff = CLASS_CUTOFFS[detected_class]
    mismatch_penalty = calc_mismatch_penalty(detected_cutoff, expected_cutoff)
    gradient_penalty = calc_gradient_penalty(gradient)
    final_confidence = max(0.1, classifier_conf - mismatch_penalty - gradient_penalty)

    # 6. Generate warnings
    warnings = []
    if mismatch_penalty > 0.15:
        warnings.append(f"Cutoff mismatch: expected {expected_cutoff}, found {detected_cutoff}")
    if gradient_penalty > 0.15:
        warnings.append("Gradual frequency rolloff - may be natural, not encoded")

    return AnalysisResult(...)
```

**Examples:**

| File | Stated | Detected | Transcode? | Gap |
|------|--------|----------|------------|-----|
| song.flac (1411 kbps) | LOSSLESS | 128 | Yes | 6 |
| song.mp3 (320 kbps) | 320 | 192 | Yes | 3 |
| song.mp3 (320 kbps) | 320 | 320 | No | 0 |
| song.mp3 (192 kbps) | 192 | 320 | No | -1 |

## Output Structure

```python
@dataclass
class AnalysisResult:
    filename: str
    file_format: str              # Container: "mp3", "flac", "wav"
    stated_class: str             # What file claims: "320", "LOSSLESS"
    detected_class: str           # What content is: "192", "V0"
    detected_cutoff: int          # Frequency in Hz
    is_transcode: bool            # stated_rank > detected_rank
    quality_gap: int              # Severity: 0-6
    confidence: float             # After penalties: 0.0-1.0
    warnings: List[str]           # Discrepancy messages
```

## Beets Integration

Files are routed based on transcode detection:

```python
if is_transcode:
    # Route to transcode folder, optionally sub-sorted by severity
    move_to(f"transcodes/{quality_gap}/")
else:
    # Route to main library, organized by quality
    move_to(f"library/{detected_class}/")
```

## Key Design Decisions

1. **7 classes**: 128, V2, 192, V0, 256, 320, LOSSLESS
2. **Hybrid detection**: Classifier primary, cutoff detector validates
3. **Band ratio comparison**: Sliding window for robustness
4. **Coarse-to-fine scanning**: Efficient frequency detection
5. **Gradient sharpness**: Distinguishes artificial vs natural rolloff
6. **Quality ranking**: Transcode = stated rank > detected rank
7. **Confidence penalties**: Mismatches reduce confidence transparently
8. **Quality gap**: Measures transcode severity for sorting/filtering
