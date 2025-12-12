# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
uv run python -c "import beetsplug.bitrater"
uv run beet bitrater --help
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=beetsplug.bitrater

# Run specific test file
uv run pytest tests/test_analyzer.py
uv run pytest tests/test_classifier.py
uv run pytest tests/test_spectrum.py
```

### Code Quality
```bash
# Format code
uv run black beetsplug/ tests/

# Sort imports
uv run isort beetsplug/ tests/

# Lint code
uv run ruff check beetsplug/ tests/

# Lint and fix
uv run ruff check --fix beetsplug/ tests/
```

### Plugin Usage
```bash
# Analyze audio files
uv run beet bitrater [query]

# Train classifier with known-good files
uv run beet bitrater --train --save-model models/trained_model.pkl

# Verbose analysis output
uv run beet bitrater -v [query]
```

## Architecture Overview

### Core Components

**Plugin Entry Point** (`beetsplug/bitrater/plugin.py`):
- `BitraterPlugin`: Main beets plugin class that integrates with beets library
- Handles CLI commands, configuration, and database field registration
- Manages parallel processing of audio files
- Provides auto-analysis during import if enabled

**Audio Analysis Pipeline**:
1. **FileAnalyzer** (`file_analyzer.py`): Extracts metadata from audio files
2. **SpectrumAnalyzer** (`spectrum.py`): Performs spectral analysis and feature extraction
3. **QualityClassifier** (`classifier.py`): SVM-based classification of audio quality
4. **AudioQualityAnalyzer** (`analyzer.py`): Orchestrates the analysis pipeline

### Key Features

**Spectral Analysis**:
- Frequency band analysis using 256 mel-scale bands
- Detection of compression artifacts and frequency cutoffs
- Ultrasonic content analysis for lossless detection
- Statistical features: spectral flatness, rolloff, contrast

**Machine Learning Classification**:
- SVM classifier with RBF kernel for bitrate prediction
- Supports CBR bitrates: 128, 160, 192, 224, 256, 320 kbps
- VBR preset detection: V0, V2, V4, V6
- Lossless vs lossy format classification

**Transcoding Detection**:
- Compares stated bitrate vs detected original bitrate
- Identifies lossless files transcoded from lossy sources
- Metadata consistency checking
- Confidence scoring for all predictions

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

Training data should be organized as:
```
training_data/
├── 128/          # 128 kbps CBR files
├── 320/          # 320 kbps CBR files
├── v0/           # V0 VBR files
├── lossless/     # FLAC/WAV files
└── encoded/      # Transcoded test files
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
- SVM configuration: **Polynomial kernel** with degree=2, γ=1, C=1
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

This extends the paper's 5-class problem (128, 192, 224, 256, 320) by adding VBR presets and lossless detection, which are critical for real-world music library analysis.

### Feature Set (171 total features)

**Power Spectral Density (150 features):**
- 150 mel-scale bands covering 16-22 kHz frequency range
- Captures compression artifacts and frequency cutoffs

**Cutoff Detection (6 features):**
- Encoder-agnostic frequency transition detection
- Identifies sharp rolloffs characteristic of lossy encoders

**Temporal Features (8 features):**
- VBR vs CBR discrimination
- Bitrate variability analysis

**Artifact Features (6 features):**
- Transcoding detection
- Compression artifact quantification

**Format Flag (1 feature):**
- Binary is_vbr indicator

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
- VBR preset confusion (V2 ↔ 192, V0 ↔ 256)
- LOSSLESS vs 320 kbps separation (requires 20-22 kHz analysis)
- Limited training data for middle-quality classes

The polynomial kernel (d=2) is specifically chosen to capture quadratic feature interactions between PSD bands, which better models how frequency cutoffs manifest compared to the more flexible but overfitting-prone RBF kernel.

### Testing

Test files are organized under `tests/`:
- `test_analyzer.py`: End-to-end analysis testing
- `test_classifier.py`: Machine learning model testing
- `test_spectrum.py`: Spectral analysis testing
- `conftest.py`: Shared test fixtures
- `data/`: Test audio files (original and transcoded)

The plugin uses spectral analysis techniques from research on MP3 quality detection through frequency spectrum analysis.