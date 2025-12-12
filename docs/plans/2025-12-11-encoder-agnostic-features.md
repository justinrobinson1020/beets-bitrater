# Encoder-Agnostic Spectral Feature Extraction

**Date:** 2025-12-11
**Status:** Implementation
**Goal:** Improve classifier accuracy from 74% toward 97% by adding encoder-agnostic temporal and artifact features

## Problem Statement

Current classifier accuracy is 74% with major confusion between:
- V2 ↔ 192 (similar bitrates ~190 kbps, both have cutoffs ~18.5-19 kHz)
- V0 ↔ LOSSLESS (V0 cutoff ~19.5 kHz is close to natural rolloff)

Simple cutoff + ultrasonic ratio features don't discriminate well because:
- Per-sample normalization destroys absolute level information
- No temporal analysis (VBR varies over time, CBR doesn't)
- No artifact detection for transcode identification
- Hardcoded 16 kHz shelf is LAME-specific, not encoder-agnostic

## Design

### Feature Architecture (~170 features)

```
SpectralFeatures:
├── PSD Bands (150)              # Existing - 16-22 kHz power spectral density
├── Cutoff Features (6)          # Encoder-agnostic transition detection
│   ├── primary_cutoff           # Main frequency cutoff (normalized 0-1)
│   ├── num_transitions          # Number of significant energy drops detected
│   ├── first_transition_freq    # First energy drop location (normalized)
│   ├── first_transition_mag     # Magnitude of first drop (dB, normalized)
│   ├── cutoff_gradient          # Sharpness of primary cutoff
│   └── transition_gap           # Gap between first transition and cutoff
├── Temporal Features (8)        # NEW - VBR vs CBR discrimination
│   ├── cutoff_variance          # Std dev of cutoff over time windows
│   ├── cutoff_min               # Minimum cutoff detected (normalized)
│   ├── cutoff_max               # Maximum cutoff detected (normalized)
│   ├── cutoff_stability_ratio   # (max-min)/mean - VBR indicator
│   ├── hf_energy_variance       # Energy variance in 16-20 kHz band
│   ├── hf_energy_trend          # Linear trend of HF energy over time
│   ├── temporal_consistency     # Correlation between adjacent windows
│   └── frame_energy_variance    # Per-frame energy variance
└── Artifact Features (6)        # NEW - Transcode detection
    ├── above_cutoff_sparsity    # % of frames with energy above cutoff
    ├── above_cutoff_noise_ratio # Noise-like vs musical content
    ├── spectral_flatness_hf     # Flatness in 18-22 kHz
    ├── spectral_discontinuity   # Abrupt changes in spectrum
    ├── harmonic_residual        # Non-harmonic content ratio
    └── interpolation_score      # Transcode artifact indicator
```

### Implementation Architecture

```
Audio File → librosa.load() → STFT (time-frequency)
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              PSD Bands (150)  Temporal Analysis  Artifact Detection
                    ↓               ↓               ↓
                    └───────────────┴───────────────┘
                                    ↓
                         Feature Vector (~170)
                                    ↓
                              Classifier
```

### Key Algorithm: Encoder-Agnostic Transition Detection

Instead of hardcoding LAME's 16 kHz shelf, scan 12-22 kHz for ANY significant energy transitions:

```python
def detect_spectral_transitions(psd, freqs):
    """
    Scan 12-22 kHz for ALL significant energy drops.

    Returns list of transitions with:
    - frequency: where the drop occurs
    - magnitude: how much energy drops (dB)
    - sharpness: how abrupt the transition is

    This is encoder-agnostic - detects patterns, lets classifier learn meaning:
    - LAME: 2 transitions (shelf ~16kHz + cutoff)
    - FhG/other: 1 transition (just cutoff)
    - Lossless: 0 sharp transitions (natural rolloff)
    - Transcode: transitions + artifacts above
    """
```

### Temporal Analysis for VBR Detection

VBR files have varying cutoff frequencies frame-to-frame; CBR files have consistent cutoffs.

```python
def extract_temporal_features(stft_magnitude, sample_rate, n_windows=20):
    """
    Split STFT into n_windows, detect cutoff in each.

    VBR indicator: high cutoff_variance, high cutoff_stability_ratio
    CBR indicator: low variance, consistent cutoff
    """
```

### Artifact Detection for Transcodes

Transcodes show sparse, noisy energy above the original cutoff ("purple bars" in spectrograms).

```python
def extract_artifact_features(stft_magnitude, freqs, detected_cutoff):
    """
    Analyze content above detected cutoff.

    True lossless: Dense, musical content (if any)
    True lossy: No content above cutoff
    Transcode: Sparse, noise-like artifacts above cutoff
    """
```

## Files to Modify

1. **types.py** - Add new feature fields to SpectralFeatures
2. **spectrum.py** - Add TemporalAnalyzer class with STFT-based extraction
3. **classifier.py** - Update _extract_features() for ~170 features
4. **constants.py** - Add new feature extraction parameters
5. **tests/** - Update tests for new feature count

## Validation Plan

1. Run validation on full training set (10,402 samples)
2. Compare per-class accuracy, especially V2/192 and V0/LOSSLESS
3. Target: >85% overall, >70% on confused pairs

## Expected Outcomes

| Class    | Current | Target |
|----------|---------|--------|
| 128      | 100%    | 100%   |
| V2       | 67%     | 80%+   |
| 192      | 54%     | 75%+   |
| V0       | 41%     | 70%+   |
| 256      | 97%     | 97%+   |
| 320      | 97%     | 97%+   |
| LOSSLESS | 62%     | 80%+   |
