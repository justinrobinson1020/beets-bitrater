# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This is the **server development environment** running on CT 113 (beets container) in a Proxmox homelab.

- **Container**: LXC 113, Ubuntu 22.04, Python 3.10.12
- **Project location**: /root/beets-bitrater/
- **Training data**: /mnt/bitrater/training_data/ (NVMe-backed ZFS dataset)
- **Source FLACs**: /music/red_downloads/flac/ (mojo-dojo pool, ~2,400 files, 85GB)
- **Feature cache**: ~/.cache/bitrater/features/

### Storage Layout

| Path | Storage | Purpose |
|------|---------|---------|
| /root/beets-bitrater/ | nvme-pool (container root) | Code, venv, tests, models |
| /mnt/bitrater/training_data/encoded/ | nvme-pool (ZFS dataset) | Encoded MP3 training data |
| /music/red_downloads/flac/ | mojo-dojo (spinning disk) | Source lossless FLACs |
| ~/.cache/bitrater/features/ | nvme-pool (container root) | Feature extraction cache |

## Development Commands

### Installation and Setup (using uv)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates .venv automatically)
uv sync

# Install with dev dependencies
uv sync --all-extras

# Add a new dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>
```

### Running Commands
```bash
# Run any command in the virtual environment
uv run <command>

# Examples:
uv run pytest tests/
uv run python -c "import bitrater"
uv run bitrater --help
```

### Testing
```bash
# Run all tests
uv run python -m pytest tests/

# Run tests with coverage
uv run python -m pytest tests/ --cov=bitrater --cov=beetsplug

# Run specific test file
uv run python -m pytest tests/test_analyzer.py
uv run python -m pytest tests/test_classifier.py
uv run python -m pytest tests/test_spectrum.py
```

### Code Quality
```bash
# Format code
uv run black bitrater/ beetsplug/ tests/

# Sort imports
uv run isort bitrater/ beetsplug/ tests/

# Lint code
uv run ruff check bitrater/ beetsplug/ tests/

# Lint and fix
uv run ruff check --fix bitrater/ beetsplug/ tests/
```

### Standalone CLI
```bash
# Analyze audio files
uv run bitrater analyze <file-or-dir>

# Train classifier from encoded training data
uv run bitrater train --source-dir /mnt/bitrater/training_data/encoded --save-model models/trained_model.pkl --threads 8

# Validate accuracy
uv run bitrater validate --source-dir /mnt/bitrater/training_data/encoded

# Generate training data (transcode FLACs to MP3s at all bitrate classes)
uv run bitrater transcode --source-dir /music/red_downloads/flac --output-dir /mnt/bitrater/training_data/encoded --workers 8
```

### Beets Plugin Usage
```bash
# Analyze audio files via beets
uv run beet bitrater [query]

# Train classifier with known-good files
uv run beet bitrater --train --save-model models/trained_model.pkl

# Verbose analysis output
uv run beet bitrater -v [query]
```

## Architecture Overview

### Package Structure

The project has two packages:
- **`bitrater/`** — Standalone core library (no beets dependency). Contains all analysis, classification, and training logic. Provides the `bitrater` CLI.
- **`beetsplug/bitrater/`** — Thin beets plugin wrapper. Only contains `plugin.py` which imports from `bitrater.*`.

### Core Components

**Standalone CLI** (`bitrater/cli.py`):
- `bitrater analyze/train/validate/transcode` subcommands
- No beets dependency required

**Beets Plugin** (`beetsplug/bitrater/plugin.py`):
- `BitraterPlugin`: Beets integration — imports from `bitrater.*`
- Handles beets CLI commands, configuration, and database field registration

**Audio Analysis Pipeline** (all in `bitrater/`):
1. **FileAnalyzer** (`file_analyzer.py`): Extracts metadata from audio files
2. **SpectrumAnalyzer** (`spectrum.py`): Performs spectral analysis and feature extraction
3. **QualityClassifier** (`classifier.py`): SVM-based classification of audio quality
4. **AudioQualityAnalyzer** (`analyzer.py`): Orchestrates the analysis pipeline

**Feature Cache** (`bitrater/feature_cache.py`):
- NPZ format with SHA256 file keys
- Thread-safe with fcntl file locking
- JSON metadata (no pickle), safe to load with allow_pickle=False
- Default location: ~/.cache/bitrater/features/

**Transcoding** (`bitrater/transcode.py`):
- Generates training data by encoding FLACs to MP3 at all bitrate classes
- Skips already-processed files (safe to interrupt and resume)
- Creates lossless symlinks to source files for training

### Key Features

**Spectral Analysis**:
- Frequency band analysis using 256 mel-scale bands
- Detection of compression artifacts and frequency cutoffs
- SFB21 band analysis and spectral rolloff features
- Statistical features: spectral flatness, rolloff, contrast

**Machine Learning Classification**:
- SVM classifier with polynomial kernel (degree=2) for bitrate prediction
- Supports CBR bitrates: 128, 160, 192, 224, 256, 320 kbps
- VBR preset detection: V0, V2, V4, V6
- Lossless vs lossy format classification

**Transcoding Detection**:
- Compares stated bitrate vs detected original bitrate
- Identifies lossless files transcoded from lossy sources
- Metadata consistency checking
- Confidence scoring for all predictions

**Feature Set (193 total features):**
- 150 PSD bands + 6 cutoff + 6 SFB21 + 4 rolloff + 6 discriminative + 4 temporal + 6 crossband + 5 cutoff_cleanliness + 6 MDCT

### Database Fields

The plugin adds these fields to beets items:
- `original_bitrate`: Estimated original bitrate
- `bitrate_confidence`: Confidence score (0.0-1.0)
- `is_transcoded`: Boolean transcoding flag
- `spectral_quality`: Overall spectral quality score
- `analysis_version`: Version of analysis used
- `analysis_date`: Timestamp of analysis
- `format_warnings`: Warning messages

### Training Data Structure

Training data is organized as:
```
/mnt/bitrater/training_data/encoded/
├── lossy/
│   ├── 128/          # 128 kbps CBR MP3s
│   ├── 192/          # 192 kbps CBR
│   ├── 256/          # 256 kbps CBR
│   ├── 320/          # 320 kbps CBR
│   ├── v0/           # V0 VBR preset
│   ├── v2/           # V2 VBR preset
│   └── v4/           # V4 VBR preset
└── lossless/          # Symlinks to source FLACs
```

### Configuration Options

```yaml
bitrater:
    auto: false                    # Auto-analyze on import
    min_confidence: 0.8           # Minimum confidence threshold
    warn_transcodes: true         # Show transcode warnings
    threads: null                 # Analysis threads (null = auto)
    model_path: null              # Path to trained model
    training_dir: null            # Training data directory
```

## Methodology & Performance Baseline

### Research Foundation

This project implements MP3 bitrate detection based on the paper **"MP3 Bit Rate Quality Detection through Frequency Spectrum Analysis"** by D'Alessandro & Shi (2009). The approach uses power spectral density (PSD) analysis of high-frequency bands to detect compression artifacts characteristic of different encoding quality levels.

**Key methodological alignment:**
- Train/test split: **20% train / 80% test** (matches paper's 500/2012 sample split)
- SVM configuration: **Polynomial kernel** with degree=2, gamma=1, C=1
- Feature engineering: PSD-based frequency analysis with encoder-agnostic features

### Classification Problem

**7-class quality detection:**
1. 128 kbps CBR (lowest quality)
2. V2 VBR preset (~170-210 kbps)
3. 192 kbps CBR
4. V0 VBR preset (~220-260 kbps)
5. 256 kbps CBR
6. 320 kbps CBR (highest lossy quality)
7. LOSSLESS (FLAC/WAV)

### Current Performance Baseline

**Overall Accuracy: ~74%** (Target: 85%+)

**Per-class accuracy:**
- 128 kbps: ~100% (excellent)
- V2 VBR: ~67% (needs improvement)
- 192 kbps: ~54% (needs improvement)
- V0 VBR: ~41% (needs improvement)
- 256 kbps: ~97% (excellent)
- 320 kbps: ~97% (excellent)
- LOSSLESS: ~62% (needs improvement)

**Key challenges:**
- VBR preset confusion (V2 <-> 192, V0 <-> 256)
- LOSSLESS vs 320 kbps separation (requires 20-22 kHz analysis)
- Limited training data for middle-quality classes

### Performance Profile

- **FFT/STFT = 78%** of per-file feature extraction time
- **Audio loading = 22%** (I/O bound)
- Long file (6.7min): ~4.2s total, Short file (2.8min): ~1.6s total
- Parallelism: joblib with threading backend (analysis) or loky backend (training)
- Thread safety: aggressive clamping via env vars at module import + threadpool_limits

### Testing

Test files are organized under `tests/`:
- `test_analyzer.py`: End-to-end analysis testing
- `test_classifier.py`: Machine learning model testing
- `test_spectrum.py`: Spectral analysis testing
- `conftest.py`: Shared test fixtures
- `data/`: Test audio files (original and transcoded)

132 tests, all passing.

## Development Workflow

- Never reference claude or claude code in commit messages
- Always use bash (not zsh) on this Linux server
