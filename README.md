# beets-bitrater

Audio quality analysis and bitrate detection for MP3 files. Detects the true encoding quality of audio files using spectral analysis and machine learning, identifying transcodes (e.g., a 128 kbps MP3 re-encoded as 320 kbps).

Available as a **standalone CLI tool** or as a **[beets](https://beets.io/) plugin**.

## Features

- **7-class bitrate classification**: 128, 192, 256, 320 kbps CBR, V0/V2 VBR presets, and lossless
- **Transcode detection**: identifies files whose stated bitrate doesn't match their true encoding quality
- **Pre-trained deep learning model**: ships with a CNN+BiLSTM model achieving 98.4% accuracy
- **SVM fallback**: polynomial kernel SVM classifier (74% accuracy) for custom training
- **Confidence scoring**: every prediction includes a confidence score
- **Feature caching**: thread-safe NPZ cache avoids redundant spectral analysis
- **Parallel processing**: multi-threaded analysis via joblib

## Installation

Requires Python 3.10+ and [FFmpeg](https://ffmpeg.org/).

### Standalone (no beets)

```bash
pip install beets-bitrater
```

### With beets plugin

```bash
pip install "beets-bitrater[beets]"
```

Then enable the plugin in your beets config (`~/.config/beets/config.yaml`):

```yaml
plugins: bitrater
```

### With training support (PyTorch)

Only needed if you want to train your own models:

```bash
pip install "beets-bitrater[training]"
```

### From source (with uv)

```bash
git clone https://github.com/justinrobinson1020/beets-bitrater.git
cd beets-bitrater
uv sync              # standalone
uv sync --all-extras # with beets + training + dev dependencies
```

## Quick Start

### Standalone CLI

```bash
# Analyze a single file
bitrater analyze song.mp3

# Analyze a directory
bitrater analyze /path/to/music/

# Verbose output (show warnings)
bitrater -v analyze /path/to/music/
```

Example output:
```
[OK] song.mp3: MP3 320kbps (confidence: 95%)
[TRANSCODE] another.mp3: MP3 128kbps (confidence: 88%)
```

### Beets Plugin

```bash
# Analyze your library (or a subset via query)
beet bitrater
beet bitrater artist:radiohead

# Verbose output
beet bitrater -v
```

The plugin stores results in beets' database as custom fields:

| Field | Description |
|-------|-------------|
| `original_bitrate` | Estimated true encoding bitrate |
| `bitrate_confidence` | Confidence score (0.0-1.0) |
| `is_transcoded` | Whether the file appears to be a transcode |
| `spectral_quality` | Overall spectral quality score |
| `format_warnings` | Warning messages from analysis |

## Pre-trained Model

Bitrater ships with a pre-trained deep learning model that works out of the box. No training is required for typical use. See [MODEL_CARD.md](MODEL_CARD.md) for full details on the model architecture, training data, and performance metrics.

The bundled model achieves **98.4% accuracy** across all 7 classes on a held-out test set. To disable it and use the SVM classifier instead:

```bash
bitrater analyze --no-dl /path/to/music/
```

### Training Your Own SVM Model

If you prefer to train a custom SVM classifier on your own data:

**1. Generate training data** by transcoding lossless files to MP3:

```bash
bitrater transcode --source-dir /path/to/flacs --output-dir /path/to/training_data --workers 8
```

**2. Train and validate:**

```bash
bitrater train --source-dir /path/to/training_data --save-model models/model.pkl --threads 8
bitrater validate --source-dir /path/to/training_data
```

**3. Use the trained model:**

```bash
bitrater analyze --no-dl --model models/model.pkl /path/to/music/
```

## Beets Plugin Configuration

All options and their defaults:

```yaml
bitrater:
    auto: false              # Auto-analyze on import
    min_confidence: 0.8      # Minimum confidence threshold
    warn_transcodes: true    # Show transcode warnings
    threads: null            # Analysis threads (null = auto)
    model_path: null         # Path to trained SVM model
    training_dir: null       # Training data directory
```

## How It Works

### Spectral Analysis

Audio files are analyzed in the frequency domain. MP3 encoding introduces characteristic artifacts:

- **Frequency cutoffs**: lower bitrates have lower high-frequency cutoffs (e.g., 128 kbps cuts off around 16 kHz)
- **Spectral flatness**: lossy compression reduces spectral detail in high frequencies
- **SFB21 band**: the highest scale factor band is a strong indicator of encoding quality

### SVM Classifier (Baseline)

Based on [D'Alessandro & Shi (2009)](https://doi.org/10.1109/ICME.2009.5202733), "MP3 Bit Rate Quality Detection through Frequency Spectrum Analysis":

- 181 features: PSD bands, cutoff detection, temporal statistics, artifact metrics
- Polynomial kernel SVM (degree=2)
- ~74% overall accuracy

### Deep Learning Classifier (Primary)

Two-stage CNN + BiLSTM architecture (~1.1M total parameters):

- **Stage 1**: CNN feature extractor on dual-band spectrograms (64 mel + 64 linear HF bins, 2-second windows)
- **Stage 2**: BiLSTM with multi-head attention over sequences of 48 CNN features, plus 211 auxiliary features (SVM spectral + global modulation DCT)
- Focal loss with class weighting, file-level aggregation across all sequences
- **98.4% overall accuracy** with all classes above 96% F1

See [MODEL_CARD.md](MODEL_CARD.md) for complete details.

## Development

```bash
# Run tests
uv run python -m pytest tests/

# Run tests with coverage
uv run python -m pytest tests/ --cov=bitrater --cov=beetsplug

# Format and lint
uv run black bitrater/ beetsplug/ tests/
uv run ruff check --fix bitrater/ beetsplug/ tests/
```

## License

[MIT](LICENSE)
