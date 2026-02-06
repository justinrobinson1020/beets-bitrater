# GPU Acceleration for beets-bitrater — DEFERRED

## Decision: Defer GPU implementation

After benchmarking real feature extraction performance, GPU acceleration provides only **~1.5x marginal improvement** over server CPU parallelism — not enough to justify the complexity.

## Benchmarking Data (MacBook CPU, per file)

| Operation | Long file (6.7min) | Short file (2.8min) | % of total |
|---|---|---|---|
| Audio loading (librosa) | 949ms (22%) | 362ms (22%) | I/O bound, irreducible |
| Welch PSD (9 calls) | 2,036ms (48%) | 781ms (48%) | GPU-acceleratable |
| STFT (2 calls) | 1,260ms (30%) | 489ms (30%) | GPU-acceleratable |
| Other (gradient, medfilt, etc.) | <1ms (~0%) | <1ms (~0%) | Not worth abstracting |
| **Total** | **4,245ms** | **1,632ms** | |

## Throughput Comparison

| Setup | Effective throughput | 10k files | Marginal gain |
|---|---|---|---|
| MacBook (4 workers, throttled) | ~1 file/sec | ~3 hours | baseline |
| Server CPU (10 workers, sustained) | ~2.4 files/sec | ~70 min | 2.5x over MacBook |
| Server GPU (single file) | ~0.8 files/sec | **slower than CPU parallel** | N/A |
| Server GPU (batch of 16) | ~3-4 files/sec | ~45 min | 1.5x over server CPU |

## Why Not GPU

1. **Single-file GPU is slower than CPU parallel** — GPU wins require batch processing
2. **Batch processing is complex** — variable-length padding, OOM recovery, memory management
3. **Marginal gain** — 70 min → 45 min for 10k files (saves ~25 min per run)
4. **Reimplementing Welch's PSD** in torch with exact scipy equivalence is non-trivial
5. **PyTorch is ~2GB** optional dependency for ~1.5x throughput improvement
6. **Cross-platform fragmentation** — MPS quirks, CUDA-only features, fallback paths

## Future Extension Point

If GPU becomes justified (larger training sets, more feature development iterations), the minimal viable abstraction is:

```python
class ComputeBackend(Protocol):
    def welch_psd(self, y: np.ndarray, sr: int, nperseg: int) -> tuple[np.ndarray, np.ndarray]: ...
    def stft(self, y: np.ndarray, n_fft: int) -> np.ndarray: ...
```

Only 2 methods needed. Everything else (gradient, medfilt, linregress) stays as direct scipy/numpy calls (<1ms each). The 8 call sites in `spectrum.py` that would need to delegate:
- `spectrum.py:131-133` — welch in `analyze_file()`
- `spectrum.py:364` — welch in `_extract_temporal_features()` (x8)
- `spectrum.py:511` — stft in `_extract_sfb21_features()`
- `spectrum.py:588` — stft in `_extract_rolloff_features()`
- `spectrum.py:722` — welch in `get_psd()`

No action needed now.
