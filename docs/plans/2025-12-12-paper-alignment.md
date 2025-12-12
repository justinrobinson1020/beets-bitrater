# Paper Alignment Plan

**Date:** 2025-12-12
**Status:** Ready for implementation
**Goal:** Improve classifier accuracy from ~74% toward 97% by aligning with D'Alessandro & Shi (2009) methodology

## Background

The project implements MP3 bitrate detection based on the paper "MP3 Bit Rate Quality Detection through Frequency Spectrum Analysis" (D'Alessandro & Shi, 2009). Current accuracy is ~74%, significantly below the paper's 97%.

Analysis identified two key methodological differences causing the gap:
1. **Train/test split:** Project uses 80/20, paper uses 20/80
2. **SVM kernel:** Project uses RBF, paper uses polynomial (d=2)

## Decisions Made

### 1. Train/Test Split: 20% Train / 80% Test

The paper achieved 97% accuracy training on only 500 samples (20%) and testing on 2012 samples (80%). This prioritizes generalization testing over maximizing training data.

**Rationale:** The paper proved that well-engineered PSD features are highly discriminative - you don't need massive training data. With ~9,000 files, 20% training (~1,800 samples) is still more than the paper used.

### 2. SVM Kernel: Polynomial (d=2, γ=1, C=1)

The paper specifies: `K(xi, xj) = (γ·xi^T·xj + C)^d` with d=2, γ=1, C=1

**Rationale:** Polynomial kernel captures quadratic feature interactions, which matches how frequency cutoffs manifest in PSD bands. RBF is more flexible but prone to overfitting.

### 3. Feature Range: Keep 16-22 kHz (171 features)

The paper uses 100 bands covering 16-20 kHz. This project extends to 22 kHz with additional features.

**Rationale:** The 7-class problem (vs paper's 5-class) requires distinguishing LOSSLESS from 320 kbps, which needs the 20-22 kHz range. The extra features (cutoff, temporal, artifact) support transcode detection.

**Feature breakdown:**
- PSD Bands: 150 (16-22 kHz)
- Cutoff Features: 6 (encoder-agnostic transition detection)
- Temporal Features: 8 (VBR vs CBR discrimination)
- Artifact Features: 6 (transcode detection)
- is_vbr flag: 1
- **Total: 171 features**

### 4. Logging: Propagate Only

Remove custom handler, rely on beets' logging coordination for parallel processing sync.

### 5. Cache: Path + Mtime + Size

Replace content-based SHA-256 hash with path+mtime+size key to avoid reading entire files on cache lookup.

## Implementation Changes

### File: `beetsplug/bitrater/classifier.py`

Change SVM parameters:
```python
# Before
self.classifier = SVC(
    kernel="rbf",
    gamma="scale",
    C=1.0,
    probability=True,
    class_weight="balanced",
)

# After
self.classifier = SVC(
    kernel="poly",
    degree=2,
    gamma=1,
    C=1,
    probability=True,
    class_weight="balanced",
)
```

### File: `beetsplug/bitrater/analyzer.py`

Change default test_size:
```python
# Before
def validate_from_directory(
    self,
    training_dir: Path,
    test_size: float = 0.2,  # 20% test
    ...
)

# After
def validate_from_directory(
    self,
    training_dir: Path,
    test_size: float = 0.8,  # 80% test (paper methodology)
    ...
)
```

### File: `beetsplug/bitrater/plugin.py`

Fix logging duplication:
```python
# Before
def _enable_verbose_logging(self) -> None:
    handler_exists = any(
        getattr(h, "_bitrater_verbose", False) for h in logger.handlers
    )
    if not handler_exists:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        handler._bitrater_verbose = True
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = True

# After
def _enable_verbose_logging(self) -> None:
    """Ensure INFO-level logs emit via beets' logging system."""
    logger.setLevel(logging.INFO)
    logger.propagate = True
    # Don't add custom handler - use beets' coordinated output
```

Update help text:
```python
# Before
help="validate model accuracy with train/test split (80/20)"

# After
help="validate model accuracy with train/test split (20/80)"
```

### File: `beetsplug/bitrater/training_data/feature_cache.py`

Replace content hash with path+mtime+size:
```python
# Before
def _get_file_hash(self, file_path: Path) -> str:
    """Calculate SHA-256 hash of file content."""
    hasher = hashlib.sha256()
    buffer_size = 65536
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()

# After
def _get_file_key(self, file_path: Path) -> str:
    """Generate cache key from path, mtime, and size (no file read)."""
    stat = file_path.stat()
    key_data = f"{file_path.resolve()}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.sha256(key_data.encode()).hexdigest()
```

Also update all calls from `_get_file_hash` to `_get_file_key`.

### File: `CLAUDE.md`

Add baseline and methodology sections (see below).

## Expected Outcomes

| Class | Current | Target |
|-------|---------|--------|
| 128 | ~100% | 100% |
| V2 | ~67% | 80%+ |
| 192 | ~54% | 75%+ |
| V0 | ~41% | 70%+ |
| 256 | ~97% | 97%+ |
| 320 | ~97% | 97%+ |
| LOSSLESS | ~62% | 80%+ |
| **Overall** | **~74%** | **85%+** |

## Validation Plan

1. Clear feature cache (new cache keys incompatible)
2. Run validation: `uv run beet bitrater --train --validate --verbose`
3. Compare per-class accuracy, especially V2/192 and V0/LOSSLESS
4. If accuracy improves significantly, commit changes
5. If not, investigate polynomial kernel hyperparameters

## References

- D'Alessandro, B., & Shi, Y. Q. (2009). MP3 bit rate quality detection through frequency spectrum analysis. MM&Sec'09.
