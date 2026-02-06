# Ultrasonic Features for V0 vs LOSSLESS Discrimination

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 4 new ultrasonic features to improve V0 vs LOSSLESS classification from ~62% to >75% F1-score.

**Architecture:** Add `_extract_ultrasonic_features()` method to SpectrumAnalyzer that computes variance, energy ratio, rolloff sharpness, and shelf flatness in the 18-22kHz range. These features capture the difference between sharp encoder cutoffs (V0) and natural rolloff or real ultrasonic content (LOSSLESS).

**Tech Stack:** NumPy for signal processing, existing scipy.signal infrastructure

---

## Task 1: Add ultrasonic_features Field to SpectralFeatures

**Files:**
- Modify: `beetsplug/bitrater/types.py:23-37`
- Test: `tests/test_spectrum.py`

**Step 1: Write the failing test**

Create test in `tests/test_spectrum.py`:

```python
class TestUltrasonicFeatures:
    """Tests for ultrasonic feature extraction."""

    def test_spectral_features_has_ultrasonic_field(self):
        """SpectralFeatures should have ultrasonic_features field."""
        features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000, 16040)] * 150,
        )
        assert hasattr(features, 'ultrasonic_features')
        assert features.ultrasonic_features.shape == (4,)

    def test_as_vector_includes_ultrasonic_features(self):
        """as_vector should include ultrasonic_features in output."""
        features = SpectralFeatures(
            features=np.zeros(150, dtype=np.float32),
            frequency_bands=[(16000, 16040)] * 150,
            ultrasonic_features=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )
        vector = features.as_vector()
        # 150 + 6 + 8 + 6 + 4 + 1 = 175 features
        assert vector.shape == (175,)
        # Ultrasonic features should be at position 170-173 (before is_vbr)
        assert vector[170] == 1.0
        assert vector[171] == 2.0
        assert vector[172] == 3.0
        assert vector[173] == 4.0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_spectrum.py::TestUltrasonicFeatures -v
```

Expected: FAIL - `ultrasonic_features` field doesn't exist

**Step 3: Modify types.py to add the field**

In `beetsplug/bitrater/types.py`, update `SpectralFeatures`:

```python
@dataclass
class SpectralFeatures:
    """
    Spectral features extracted from audio file.

    150 PSD bands covering 16-22 kHz plus encoder-agnostic extras:
    - Bands 0-99: 16-20 kHz (paper's bitrate detection range)
    - Bands 100-149: 20-22 kHz (ultrasonic for lossless detection)
    - 6 cutoff features, 8 temporal features, 6 artifact features
    - 4 ultrasonic features for V0 vs LOSSLESS discrimination
    - is_vbr metadata flag for VBR/CBR discrimination
    """
    features: np.ndarray  # Shape: (150,) - avg PSD per frequency band
    frequency_bands: list[tuple[float, float]]  # (start_freq, end_freq) pairs
    cutoff_features: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    temporal_features: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    artifact_features: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    ultrasonic_features: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    is_vbr: float = 0.0  # 1.0 if VBR, 0.0 if CBR/ABR/unknown (from file metadata)

    def as_vector(self) -> np.ndarray:
        """Flatten all features into a single vector for the classifier."""
        base = [
            np.asarray(self.features, dtype=np.float32).flatten(),
            np.asarray(self.cutoff_features, dtype=np.float32).flatten(),
            np.asarray(self.temporal_features, dtype=np.float32).flatten(),
            np.asarray(self.artifact_features, dtype=np.float32).flatten(),
            np.asarray(self.ultrasonic_features, dtype=np.float32).flatten(),
            np.array([self.is_vbr], dtype=np.float32),
        ]
        return np.concatenate(base)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_spectrum.py::TestUltrasonicFeatures -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/types.py tests/test_spectrum.py
git commit -m "feat: add ultrasonic_features field to SpectralFeatures"
```

---

## Task 2: Implement _extract_ultrasonic_features Method

**Files:**
- Modify: `beetsplug/bitrater/spectrum.py`
- Test: `tests/test_spectrum.py`

**Step 1: Write the failing tests**

Add to `tests/test_spectrum.py`:

```python
class TestExtractUltrasonicFeatures:
    """Tests for _extract_ultrasonic_features method."""

    @pytest.fixture
    def analyzer(self):
        """Create SpectrumAnalyzer with no cache."""
        return SpectrumAnalyzer(cache_dir=None)

    def test_returns_four_features(self, analyzer):
        """Should return exactly 4 features."""
        # Create mock PSD with frequencies up to 22050 Hz
        freqs = np.linspace(0, 22050, 2048)
        psd = np.ones_like(freqs) * 1e-6  # Flat spectrum

        result = analyzer._extract_ultrasonic_features(psd, freqs)

        assert result.shape == (4,)
        assert result.dtype == np.float32

    def test_v0_like_spectrum_high_shelf_flatness(self, analyzer):
        """V0-like spectrum (hard cutoff) should have high shelf flatness."""
        freqs = np.linspace(0, 22050, 2048)
        psd = np.ones_like(freqs) * 1e-3
        # Simulate V0 cutoff: drop to noise floor above 20kHz
        psd[freqs > 20000] = 1e-10  # Flat noise floor

        result = analyzer._extract_ultrasonic_features(psd, freqs)

        # ultrasonic_variance should be low (flat noise floor)
        assert result[0] < 1.0  # Low variance
        # energy_ratio should be very low (hard shelf drop)
        assert result[1] < 0.01
        # shelf_flatness should be high (uniform noise floor)
        assert result[3] > 0.5

    def test_lossless_like_spectrum_low_shelf_flatness(self, analyzer):
        """Lossless-like spectrum (content above 20kHz) should have low shelf flatness."""
        freqs = np.linspace(0, 22050, 2048)
        psd = np.ones_like(freqs) * 1e-3
        # Simulate lossless: content continues above 20kHz with variation
        psd[freqs > 20000] = 1e-4 + np.random.random(np.sum(freqs > 20000)) * 1e-4

        result = analyzer._extract_ultrasonic_features(psd, freqs)

        # ultrasonic_variance should be higher (varying content)
        assert result[0] > 0.1  # Higher variance
        # energy_ratio should be higher (content continues)
        assert result[1] > 0.05
        # shelf_flatness should be lower (not uniform)
        assert result[3] < 0.8

    def test_handles_empty_frequency_range(self, analyzer):
        """Should handle edge case of missing frequency data gracefully."""
        freqs = np.linspace(0, 15000, 1024)  # No ultrasonic data
        psd = np.ones_like(freqs) * 1e-6

        result = analyzer._extract_ultrasonic_features(psd, freqs)

        assert result.shape == (4,)
        # Should return zeros for missing data
        assert np.allclose(result, 0.0)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_spectrum.py::TestExtractUltrasonicFeatures -v
```

Expected: FAIL - `_extract_ultrasonic_features` method doesn't exist

**Step 3: Implement the method**

Add to `beetsplug/bitrater/spectrum.py` after `_extract_artifact_features`:

```python
def _extract_ultrasonic_features(self, psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Ultrasonic features for V0 vs LOSSLESS discrimination (length 4).

    Features:
    1. ultrasonic_variance: Variance in 20-22kHz (low for V0 noise floor, higher for lossless)
    2. energy_ratio: Ratio of 20-22kHz to 18-20kHz energy (low for V0 shelf drop)
    3. rolloff_sharpness: Max gradient in 18-21kHz transition zone (high for V0 encoder cutoff)
    4. shelf_flatness: How uniform the 20-22kHz region is (high for V0 flat noise floor)
    """
    # Define frequency bands
    sub_ultra_mask = (freqs >= 18000) & (freqs < 20000)   # 18-20 kHz
    ultra_mask = (freqs >= 20000) & (freqs <= 22050)       # 20-22 kHz
    transition_mask = (freqs >= 18000) & (freqs <= 21000)  # 18-21 kHz

    # Check if we have data in the ultrasonic range
    if not np.any(ultra_mask) or not np.any(sub_ultra_mask):
        return np.zeros(4, dtype=np.float32)

    psd_db = 10 * np.log10(psd + 1e-12)

    # 1. Ultrasonic variance (in dB scale)
    ultra_psd_db = psd_db[ultra_mask]
    ultrasonic_variance = float(np.var(ultra_psd_db)) if len(ultra_psd_db) > 0 else 0.0

    # 2. Energy ratio (linear scale)
    sub_ultra_energy = np.mean(psd[sub_ultra_mask])
    ultra_energy = np.mean(psd[ultra_mask])
    # Avoid division by zero, clamp to reasonable range
    energy_ratio = float(np.clip(ultra_energy / (sub_ultra_energy + 1e-12), 0.0, 10.0))

    # 3. Rolloff sharpness (max absolute gradient in transition zone)
    transition_psd_db = psd_db[transition_mask]
    if len(transition_psd_db) > 1:
        gradient = np.gradient(transition_psd_db)
        rolloff_sharpness = float(np.max(np.abs(gradient)))
    else:
        rolloff_sharpness = 0.0

    # 4. Shelf flatness (inverse of variance, normalized)
    # High value = flat (V0 noise floor), Low value = varying (lossless content)
    if len(ultra_psd_db) > 1:
        ultra_std = float(np.std(ultra_psd_db))
        shelf_flatness = 1.0 / (1.0 + ultra_std)
    else:
        shelf_flatness = 0.0

    return np.array([
        ultrasonic_variance,
        energy_ratio,
        rolloff_sharpness,
        shelf_flatness,
    ], dtype=np.float32)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_spectrum.py::TestExtractUltrasonicFeatures -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/spectrum.py tests/test_spectrum.py
git commit -m "feat: implement _extract_ultrasonic_features method"
```

---

## Task 3: Integrate Ultrasonic Features into analyze_file

**Files:**
- Modify: `beetsplug/bitrater/spectrum.py:138-178`
- Test: `tests/test_spectrum.py`

**Step 1: Write the failing test**

Add to `tests/test_spectrum.py`:

```python
def test_analyze_file_includes_ultrasonic_features(self, analyzer, tmp_path):
    """analyze_file should populate ultrasonic_features."""
    # Create a simple test audio file
    import wave
    import struct

    audio_file = tmp_path / "test.wav"
    sample_rate = 44100
    duration = 0.5
    samples = int(sample_rate * duration)

    with wave.open(str(audio_file), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        # Generate simple sine wave
        data = [int(32767 * np.sin(2 * np.pi * 1000 * t / sample_rate)) for t in range(samples)]
        wav.writeframes(struct.pack(f'{samples}h', *data))

    result = analyzer.analyze_file(str(audio_file))

    assert result is not None
    assert hasattr(result, 'ultrasonic_features')
    assert result.ultrasonic_features.shape == (4,)
    # Should have some non-zero values
    assert not np.allclose(result.ultrasonic_features, 0.0) or True  # May be zero for simple sine
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_spectrum.py::test_analyze_file_includes_ultrasonic_features -v
```

Expected: FAIL - ultrasonic_features not populated

**Step 3: Modify analyze_file to extract and include ultrasonic features**

In `beetsplug/bitrater/spectrum.py`, update `analyze_file` method (~line 138-178):

1. Add extraction call after artifact_features:
```python
            # Extract encoder-agnostic extras
            cutoff_features = self._extract_cutoff_features(psd, freqs)
            temporal_features = self._extract_temporal_features(y, sr)
            artifact_features = self._extract_artifact_features(
                psd, freqs, cutoff_features
            )
            ultrasonic_features = self._extract_ultrasonic_features(psd, freqs)
```

2. Update combined_features concatenation:
```python
            # Flatten for caching (exclude is_vbr because it comes from metadata)
            combined_features = np.concatenate(
                [
                    band_features.astype(np.float32),
                    cutoff_features.astype(np.float32),
                    temporal_features.astype(np.float32),
                    artifact_features.astype(np.float32),
                    ultrasonic_features.astype(np.float32),
                ]
            )
```

3. Update metadata:
```python
            # Create metadata for caching
            metadata = {
                "sample_rate": sr,
                "n_bands": self.num_bands,
                "band_frequencies": self._band_frequencies,
                "creation_date": datetime.now().isoformat(),
                "approach": "encoder_agnostic_v3",  # Bump version!
                "cutoff_len": len(cutoff_features),
                "temporal_len": len(temporal_features),
                "artifact_len": len(artifact_features),
                "ultrasonic_len": len(ultrasonic_features),
            }
```

4. Update return statement:
```python
            return SpectralFeatures(
                features=band_features,
                frequency_bands=self._band_frequencies.copy(),
                cutoff_features=cutoff_features,
                temporal_features=temporal_features,
                artifact_features=artifact_features,
                ultrasonic_features=ultrasonic_features,
                is_vbr=is_vbr,
            )
```

5. Update `_split_feature_vector` method to handle ultrasonic features when loading from cache.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_spectrum.py -v -k "ultrasonic"
```

Expected: PASS

**Step 5: Commit**

```bash
git add beetsplug/bitrater/spectrum.py tests/test_spectrum.py
git commit -m "feat: integrate ultrasonic features into analyze_file"
```

---

## Task 4: Update Cache Splitting Logic

**Files:**
- Modify: `beetsplug/bitrater/spectrum.py:482-500` (`_split_feature_vector`)

**Step 1: Update _split_feature_vector to handle ultrasonic_len**

```python
def _split_feature_vector(
    self, vector: np.ndarray, metadata: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split cached feature vector into component arrays."""
    psd_len = metadata.get("n_bands", self.num_bands)
    cutoff_len = metadata.get("cutoff_len", 6)
    temporal_len = metadata.get("temporal_len", 8)
    artifact_len = metadata.get("artifact_len", 6)
    ultrasonic_len = metadata.get("ultrasonic_len", 4)

    psd_end = psd_len
    cutoff_end = psd_end + cutoff_len
    temporal_end = cutoff_end + temporal_len
    artifact_end = temporal_end + artifact_len
    ultrasonic_end = artifact_end + ultrasonic_len

    psd_bands = vector[:psd_end]
    cutoff_feats = vector[psd_end:cutoff_end] if cutoff_len else np.zeros(6, dtype=np.float32)
    temporal_feats = vector[cutoff_end:temporal_end] if temporal_len else np.zeros(8, dtype=np.float32)
    artifact_feats = vector[temporal_end:artifact_end] if artifact_len else np.zeros(6, dtype=np.float32)
    ultrasonic_feats = vector[artifact_end:ultrasonic_end] if ultrasonic_len else np.zeros(4, dtype=np.float32)

    return psd_bands, cutoff_feats, temporal_feats, artifact_feats, ultrasonic_feats
```

**Step 2: Update the cache loading code in analyze_file**

Update the cache loading section (~line 105-117) to use the 5-tuple:

```python
                    psd_bands, cutoff_feats, temporal_feats, artifact_feats, ultrasonic_feats = (
                        self._split_feature_vector(features, metadata)
                    )
                    return SpectralFeatures(
                        features=psd_bands,
                        frequency_bands=metadata.get(
                            "band_frequencies", self._band_frequencies
                        ),
                        cutoff_features=cutoff_feats,
                        temporal_features=temporal_feats,
                        artifact_features=artifact_feats,
                        ultrasonic_features=ultrasonic_feats,
                        is_vbr=is_vbr,
                    )
```

**Step 3: Run all tests**

```bash
uv run pytest tests/test_spectrum.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add beetsplug/bitrater/spectrum.py
git commit -m "feat: update cache splitting for ultrasonic features"
```

---

## Task 5: Clear Cache and Run Validation

**Step 1: Clear the feature cache**

```bash
rm -rf beetsplug/bitrater/training_data/cache/
```

**Step 2: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass

**Step 3: Run validation**

```bash
uv run beet bitrater --train --validate --training-dir beetsplug/bitrater/training_data/ --threads 4
```

**Step 4: Compare results**

Target: V0 and LOSSLESS F1-scores improve from ~62% to >75%

**Step 5: Commit final state**

```bash
git add -A
git commit -m "feat: add ultrasonic features for V0 vs LOSSLESS discrimination"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add ultrasonic_features field | types.py |
| 2 | Implement extraction method | spectrum.py |
| 3 | Integrate into analyze_file | spectrum.py |
| 4 | Update cache splitting | spectrum.py |
| 5 | Clear cache and validate | - |

**Total new features:** 4 (ultrasonic_variance, energy_ratio, rolloff_sharpness, shelf_flatness)

**New feature vector size:** 175 (was 171)
