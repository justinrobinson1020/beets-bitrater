# Beets-Bitrater Plugin Implementation Plan

## Plugin Goals vs. Paper Methodology

### What This Plugin Does (Core Features)

| Feature | Description | User Value |
|---------|-------------|------------|
| **Bitrate Detection** | Identify original encoding bitrate of MP3s | Know true quality of lossy files |
| **Lossless Verification** | Detect genuine lossless audio (FLAC/WAV with full spectrum) | Verify lossless files are authentic |
| **Transcode Detection** | Identify "fake" lossless (e.g., FLAC encoded from 128 kbps MP3) | Catch fraudulent/mistaken transcodes |

### How the Paper Helps (Foundation)

The D'Alessandro & Shi paper provides a **proven methodology** for bitrate detection:

| Aspect | Paper's Approach |
|--------|------------------|
| **Dataset** | 2,512 songs × 5 bitrates = 12,560 samples |
| **Features** | 100 PSD bands in 16-20 kHz range |
| **Classifier** | SVM, polynomial kernel (d=2, γ=1, C=1) |
| **Accuracy** | 97% for 5-class bitrate classification |
| **Transcode Finding** | 99.3% of 128→320 transcodes correctly identified |

### Key Insight: Why Paper's Method Enables Our Features

The paper discovered that **lossy compression leaves a permanent spectral signature**:
- 128 kbps cuts frequencies above ~16 kHz
- 192 kbps cuts above ~17-18 kHz
- 256 kbps cuts above ~19 kHz
- 320 kbps cuts above ~20 kHz
- Lossless preserves full spectrum to 22.05 kHz (Nyquist)

**This signature persists through transcoding.** A FLAC made from a 128 kbps MP3 still has the 16 kHz cutoff.

### Our Extension: Adding Lossless as a Class

| Paper | This Plugin |
|-------|-------------|
| 5 classes (128, 192, 256, 320, VBR-0) | 6 classes (+Lossless) |
| 16-20 kHz analysis | 16-22 kHz analysis |
| 100 frequency bands | 150 frequency bands |
| Detects MP3 bitrate | Detects bitrate + lossless + transcodes |

---

## Current Codebase Assessment

| Component | Paper | Current Code | Status |
|-----------|-------|--------------|--------|
| Frequency range | 16-20 kHz | 16-20 kHz | ⚠️ Needs extension to 22 kHz |
| Number of bands | 100 | 100 | ⚠️ Needs extension to 150 |
| PSD calculation | Entire song | scipy.signal.welch | ✅ Correct |
| SVM kernel | poly, d=2 | poly (degree not passed!) | ❌ Bug |
| SVM gamma/C | γ=1, C=1 | γ=1, C=1 | ✅ Correct |
| Lossless detection | N/A | Broken heuristics | ❌ Needs rebuild |
| Transcode detection | N/A | Broken logic | ❌ Needs rebuild |
| Training data | 2,512 songs | ~0 songs | ❌ Critical gap |
| Extra features | None | MFCCs, spectral stats | ⚠️ Remove (unused/broken) |

### Critical Bugs Preventing Execution

1. **Missing methods**: `AudioQualityAnalyzer.analyze_file()` and `train()` not implemented
2. **Missing dataclass fields**: `FileMetadata.channels`, `SpectralFeatures.spectral_stats`
3. **SVM misconfigured**: `degree=2` parameter not passed to SVC
4. **Broken features**: MFCCs, spectral stats referenced but not properly integrated
5. **Empty tests**: All test files are empty

---

## Implementation Strategy

### Principle: Paper Foundation + Minimal Lossless Extension

We will:
1. **Implement the paper's proven methodology exactly** (100 bands, 16-20 kHz, poly SVM)
2. **Add 50 ultrasonic bands** (20-22 kHz) for lossless detection
3. **Train with 6 classes** (5 lossy + lossless)
4. **Remove broken extras** (MFCCs, spectral stats, complex heuristics)

This gives us a clean, working system with all three core features.

---

## Phase 1: Core Implementation

### 1.1: Simplify Data Types

**File:** `beetsplug/bitrater/types.py`

```python
@dataclass
class SpectralFeatures:
    """
    150 PSD bands covering 16-22 kHz:
    - Bands 0-99: 16-20 kHz (paper's bitrate detection range)
    - Bands 100-149: 20-22 kHz (ultrasonic for lossless detection)
    """
    features: np.ndarray  # Shape: (150,)
    frequency_bands: List[Tuple[float, float]]


@dataclass
class AnalysisResult:
    """Result of analyzing an audio file."""
    filename: str
    original_format: str          # "128", "192", "256", "320", "V0", "LOSSLESS"
    original_bitrate: int         # 128, 192, 256, 320, 245 (V0 avg), or 1411 (CD)
    confidence: float             # SVM decision confidence
    file_format: str              # Actual container: "mp3", "flac", "wav"
    stated_bitrate: Optional[int] # What the file claims to be
    is_transcode: bool            # True if lossless container but lossy content
    transcoded_from: Optional[str]  # e.g., "128" if FLAC contains 128 kbps content
```

**Remove:** `mfccs`, `spectral_flatness`, `spectral_rolloff`, `spectral_stats`, `ultrasonic_power`, `is_likely_lossless`, `QualityAnalysis`

---

### 1.2: Update Constants

**File:** `beetsplug/bitrater/constants.py`

```python
# Spectral analysis - extended for lossless detection
SPECTRAL_PARAMS = {
    "min_freq": 16000,   # Paper's starting frequency
    "max_freq": 22050,   # Extended to Nyquist for lossless detection
    "num_bands": 150,    # 100 (paper) + 50 (ultrasonic extension)
    "fft_size": 8192,
}

# Classification classes
BITRATE_CLASSES = {
    0: ("128", 128),      # CBR 128 kbps
    1: ("192", 192),      # CBR 192 kbps
    2: ("256", 256),      # CBR 256 kbps
    3: ("320", 320),      # CBR 320 kbps
    4: ("V0", 245),       # VBR-0 (avg ~245 kbps)
    5: ("LOSSLESS", 1411), # Lossless (CD bitrate)
}

# SVM parameters from D'Alessandro & Shi paper
CLASSIFIER_PARAMS = {
    "kernel": "poly",
    "degree": 2,       # d=2 from paper - CRITICAL
    "gamma": 1,        # γ=1 from paper
    "C": 1,            # C=1 from paper
    "coef0": 1,        # Standard for polynomial kernel
    "probability": True,  # For confidence scores
    "random_state": 42,
}

# Fix garbled line
VBR_PRESETS = [0]  # Only V0 for now
```

---

### 1.3: Fix Spectrum Analyzer

**File:** `beetsplug/bitrater/spectrum.py`

Key changes:
1. Extend frequency range to 16-22 kHz
2. Generate 150 bands instead of 100
3. Remove MFCC and spectral stat calculations
4. Keep only PSD band features

```python
def analyze_file(self, file_path: str) -> Optional[SpectralFeatures]:
    """Extract 150 PSD band features (16-22 kHz)."""
    # ... load audio with librosa ...

    # Compute PSD using scipy.signal.welch (matches paper)
    freqs, psd = signal.welch(y, sr, nperseg=self.fft_size)

    # Extract 150 band features
    band_features = self._analyze_frequency_bands_psd(psd, freqs)

    return SpectralFeatures(
        features=band_features,  # Shape: (150,)
        frequency_bands=self._band_frequencies
    )
```

**Remove:** `_analyze_frequency_bands()` (STFT version), MFCC code, spectral flatness/rolloff

---

### 1.4: Fix Classifier

**File:** `beetsplug/bitrater/classifier.py`

```python
class QualityClassifier:
    def __init__(self):
        self.classifier = SVC(
            kernel='poly',
            degree=2,      # CRITICAL: Was missing!
            gamma=1,
            C=1,
            coef0=1,
            probability=True,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False

    def _extract_features(self, spectral: SpectralFeatures) -> np.ndarray:
        """Extract the 150 PSD band features."""
        return spectral.features  # That's it - just the PSD bands

    def predict(self, features: SpectralFeatures) -> Prediction:
        """Classify audio and return prediction."""
        X = self._extract_features(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        class_idx = self.classifier.predict(X_scaled)[0]
        proba = self.classifier.predict_proba(X_scaled)[0]

        format_name, bitrate = BITRATE_CLASSES[class_idx]
        confidence = proba[class_idx]

        return Prediction(
            format_type=format_name,
            estimated_bitrate=bitrate,
            confidence=confidence
        )
```

**Remove:** Complex VBR heuristics, spectral_stats access, quality scoring

---

### 1.5: Implement Analyzer Orchestration

**File:** `beetsplug/bitrater/analyzer.py`

```python
class AudioQualityAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        self.spectrum_analyzer = SpectrumAnalyzer()
        self.file_analyzer = FileAnalyzer()
        self.classifier = QualityClassifier()

        if model_path and Path(model_path).exists():
            self.classifier.load_model(model_path)

    def analyze_file(self, file_path: str) -> Optional[AnalysisResult]:
        """Analyze a single audio file."""
        path = Path(file_path)

        # 1. Get file metadata
        metadata = self.file_analyzer.analyze_metadata(file_path)
        if not metadata:
            return None

        # 2. Extract spectral features
        features = self.spectrum_analyzer.analyze_file(file_path)
        if not features:
            return None

        # 3. Classify
        prediction = self.classifier.predict(features)

        # 4. Determine if transcoded
        file_format = path.suffix.lower().lstrip('.')
        is_lossless_container = file_format in ('flac', 'wav', 'alac', 'ape')
        detected_lossy = prediction.format_type != "LOSSLESS"

        is_transcode = is_lossless_container and detected_lossy
        transcoded_from = prediction.format_type if is_transcode else None

        return AnalysisResult(
            filename=str(path),
            original_format=prediction.format_type,
            original_bitrate=prediction.estimated_bitrate,
            confidence=prediction.confidence,
            file_format=file_format,
            stated_bitrate=metadata.bitrate,
            is_transcode=is_transcode,
            transcoded_from=transcoded_from
        )

    def train(self, training_data: Dict[str, int]) -> None:
        """Train classifier from {file_path: class_label} mapping."""
        features_list = []
        labels = []

        for path, label in training_data.items():
            features = self.spectrum_analyzer.analyze_file(path)
            if features:
                features_list.append(features)
                labels.append(label)

        self.classifier.train(features_list, labels)
```

---

### 1.6: Fix Plugin Integration

**File:** `beetsplug/bitrater/plugin.py`

1. Add `channels: int` to FileMetadata dataclass
2. Fix thread-safe progress counter (use `threading.Lock`)
3. Register import listener properly
4. Fix null check for training_dir

---

### 1.7: Training Data Structure

```
training_data/
├── lossless/        # Original FLAC/WAV files (class 5)
├── 128/             # Encoded at 128 kbps (class 0)
├── 192/             # Encoded at 192 kbps (class 1)
├── 256/             # Encoded at 256 kbps (class 2)
├── 320/             # Encoded at 320 kbps (class 3)
└── v0/              # Encoded with LAME -V 0 (class 4)
```

**Transcoding script** (`training_data/transcode.py`):
```bash
# For each lossless source in lossless/:
ffmpeg -i input.flac -acodec libmp3lame -b:a 128k 128/output.mp3
ffmpeg -i input.flac -acodec libmp3lame -b:a 192k 192/output.mp3
ffmpeg -i input.flac -acodec libmp3lame -b:a 256k 256/output.mp3
ffmpeg -i input.flac -acodec libmp3lame -b:a 320k 320/output.mp3
ffmpeg -i input.flac -acodec libmp3lame -q:a 0 v0/output.mp3
```

---

### 1.8: Write Tests

**File:** `tests/test_pipeline.py`

```python
def test_feature_extraction_shape():
    """Features should be exactly 150 values (100 paper + 50 ultrasonic)."""
    features = spectrum_analyzer.analyze_file(test_file)
    assert features.features.shape == (150,)

def test_lossless_detection():
    """True lossless file should be classified as LOSSLESS."""
    result = analyzer.analyze_file(lossless_flac)
    assert result.original_format == "LOSSLESS"
    assert result.is_transcode == False

def test_transcode_detection():
    """FLAC from 128 kbps should be detected as transcode."""
    result = analyzer.analyze_file(fake_lossless_flac)
    assert result.is_transcode == True
    assert result.transcoded_from == "128"
    assert result.file_format == "flac"

def test_mp3_bitrate_detection():
    """MP3 files should have correct bitrate detected."""
    result = analyzer.analyze_file(mp3_320)
    assert result.original_format == "320"
    assert result.original_bitrate == 320
```

---

## Phase 2: Optional Enhancements

Only pursue after Phase 1 achieves ≥85% accuracy.

| Enhancement | Description |
|-------------|-------------|
| VBR expansion | Add V2, V4, V6 classes |
| Quality scoring | Overall quality metric combining multiple factors |
| Encoder detection | Identify LAME vs FhG vs other encoders |
| Confidence intervals | Report uncertainty ranges |
| Batch optimization | Parallel processing improvements |

---

## Training Data Requirements

| Dataset Size | Expected Accuracy | Recommendation |
|--------------|-------------------|----------------|
| 100 songs | ~70-80% | Too small |
| 250 songs | ~85-90% | Minimum viable |
| 500 songs | ~90-95% | Good |
| 1000+ songs | ~95-97% | Paper-level |

With 500 source lossless songs:
- 500 lossless samples (class 5)
- 500 × 5 = 2,500 encoded MP3s (classes 0-4)
- **Total: 3,000 training samples**

---

## Files to Modify

| File | Changes |
|------|---------|
| `types.py` | Simplify to 2 dataclasses, remove broken fields |
| `constants.py` | Extend to 150 bands, fix VBR_PRESETS, add BITRATE_CLASSES |
| `spectrum.py` | Extend to 22 kHz, remove MFCC/stats, simplify |
| `classifier.py` | Fix SVM degree=2, simplify feature extraction |
| `analyzer.py` | Implement `analyze_file()` and `train()` |
| `plugin.py` | Fix threading, add channels field, register listener |
| `tests/` | Write actual tests |

---

## Success Criteria

1. ✅ `beet bitrater` runs without crashes
2. ✅ Training completes on 6-class dataset
3. ✅ ≥85% accuracy on held-out test set
4. ✅ **Lossless detection**: True FLAC → "LOSSLESS"
5. ✅ **Transcode detection**:
   - FLAC from 128 kbps → `is_transcode=True, transcoded_from="128"`
   - FLAC from 320 kbps → `is_transcode=True, transcoded_from="320"`
6. ✅ **Bitrate detection**: MP3 320 → "320", MP3 128 → "128"

---

## Execution Order

```
1.1-1.2: Types + Constants ──┐
                             ├──► 1.3: Spectrum Analyzer
                             │            │
                             │            ▼
                             └──► 1.4: Classifier
                                          │
                                          ▼
                                    1.5: Analyzer
                                          │
                                          ▼
                                    1.6: Plugin
                                          │
                                          ▼
                                    1.7: Training Data (user task)
                                          │
                                          ▼
                                    1.8: Tests + Validation
```
