# Replace ProcessPoolExecutor with joblib.Parallel

**Date:** 2025-01-12
**Status:** Approved
**Problem:** Kernel panics from thread oversubscription (800+ threads per worker)

## Problem Statement

The plugin uses `ProcessPoolExecutor` for parallel feature extraction during training. Each worker process spawns 800+ internal threads from numba/OpenMP/BLAS, causing macOS kernel panics from oversubscription.

**Current state:** 8 workers × 800 threads = 6400 threads → kernel panic
**Target state:** 8 workers × 2 threads = 16 threads → stable

## Solution

Replace `ProcessPoolExecutor` and `ThreadPoolExecutor` with `joblib.Parallel` using the `loky` backend.

### Why joblib/loky

From scikit-learn documentation:
> "Starting from joblib >= 0.14, when the loky backend is used (which is the default), joblib will tell its child processes to limit the number of threads they can use, so as to avoid oversubscription. In practice the heuristic that joblib uses is to tell the processes to use `max_threads = n_cpus // n_jobs`."

This is the battle-tested approach used by scikit-learn internally.

## Changes

### 1. analyzer.py - Training/Validation

Replace ProcessPoolExecutor with joblib.Parallel:

```python
# Before
with ProcessPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(_extract_features_worker, path): path for path in paths}
    for future in futures:
        result = future.result()

# After
from joblib import Parallel, delayed

results = Parallel(n_jobs=workers, backend="loky")(
    delayed(_extract_features_worker)(path) for path in paths
)
```

Simplify `_extract_features_worker()` by removing all env var and threadpoolctl hacks:

```python
def _extract_features_worker(file_path: str) -> tuple[str, SpectralFeatures | None]:
    """Worker function for parallel feature extraction."""
    try:
        analyzer = SpectrumAnalyzer()
        file_analyzer = FileAnalyzer()
        metadata = file_analyzer.analyze(file_path)
        is_vbr = 1.0 if metadata and metadata.encoding_type == "VBR" else 0.0
        features = analyzer.analyze_file(file_path, is_vbr=is_vbr)
        return (file_path, features)
    except Exception as e:
        logger.warning(f"Failed to extract features from {file_path}: {e}")
        return (file_path, None)
```

### 2. plugin.py - Analysis

Replace ThreadPoolExecutor with joblib.Parallel:

```python
from joblib import Parallel, delayed

def _analyze_items(self, items: Sequence[Item], thread_count: int) -> list[AnalysisResult | None]:
    """Analyze multiple items in parallel using joblib."""

    def analyze_single(item: Item) -> AnalysisResult | None:
        try:
            return self.analyzer.analyze_file(str(item.path))
        except Exception as e:
            logger.error(f"Error analyzing {item.path}: {e}")
            return None

    results = Parallel(n_jobs=thread_count, backend="loky")(
        delayed(analyze_single)(item) for item in items
    )

    return results
```

### 3. Cleanup - Remove Hacks

**`__init__.py`** - Remove env var settings, revert to simple import:
```python
"""Beets plugin for detecting original MP3 bitrate using spectral analysis."""
from beetsplug.bitrater.plugin import BitraterPlugin
__all__ = ['BitraterPlugin']
```

**`spectrum.py`** - Remove:
- `os.environ` settings for NUMBA_NUM_THREADS, OMP_NUM_THREADS, etc.
- `import numba` and `numba.set_num_threads(1)`

**`analyzer.py`** - Remove:
- Module-level env var settings
- threadpoolctl imports and usage in worker function

### 4. Tests

Update `test_parallel_training.py`:
- Change mocks from `ProcessPoolExecutor` to `joblib.Parallel`
- Update assertions for new return patterns

## Configuration

Existing configuration preserved (backwards compatible):
- `--threads` CLI flag → maps to joblib `n_jobs`
- `threads` config option → maps to joblib `n_jobs`

## Error Handling

- Workers return `(file_path, None)` on failure (unchanged)
- joblib propagates exceptions from workers
- loky is resilient to worker crashes

## Implementation Order

1. Update `analyzer.py` (training - the crash source)
2. Update `plugin.py` (analysis)
3. Clean up `__init__.py` and `spectrum.py`
4. Update tests
5. Run full test suite
6. Manual test with real files

## Risk Assessment

**Risk: Low**
- joblib is already a dependency (via scikit-learn)
- Same semantics as current code
- Battle-tested in scientific Python ecosystem
- No user-facing API changes
