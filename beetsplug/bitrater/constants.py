"""Constants used throughout the bitrater plugin."""

# Bitrate classifications
CBR_BITRATES = [128, 192, 256, 320]  # kbps
VBR_PRESETS = [0, 2]  # VBR preset levels (V0 and V2)

# VBR bitrate ranges and averages
VBR_RANGES = {
    0: (220, 260, 245),  # (min, max, average) in kbps - V0
    2: (170, 210, 190),  # V2
    4: (140, 185, 165),  # V4
}

# Classification classes (7 classes)
BITRATE_CLASSES = {
    0: ("128", 128),      # CBR 128 kbps - cutoff ~16 kHz
    1: ("V2", 190),       # VBR-2 (avg ~190 kbps) - cutoff ~18.5 kHz
    2: ("192", 192),      # CBR 192 kbps - cutoff ~19 kHz
    3: ("V0", 245),       # VBR-0 (avg ~245 kbps) - cutoff ~19.5 kHz
    4: ("256", 256),      # CBR 256 kbps - cutoff ~20 kHz
    5: ("320", 320),      # CBR 320 kbps - cutoff ~20.5 kHz
    6: ("LOSSLESS", 1411),  # Lossless (CD quality) - cutoff >21.5 kHz
}

# Reverse lookup: format name to class index
CLASS_LABELS = {name: idx for idx, (name, _) in BITRATE_CLASSES.items()}

# Quality ranking for transcode detection (higher = better quality)
QUALITY_RANK = {name: idx for idx, (name, _) in BITRATE_CLASSES.items()}
# Result: {"128": 0, "V2": 1, "192": 2, "V0": 3, "256": 4, "320": 5, "LOSSLESS": 6}

# Expected cutoff frequencies (Hz) for each class
CLASS_CUTOFFS = {
    "128": 16000,
    "V2": 18500,
    "192": 19000,
    "V0": 19500,
    "256": 20000,
    "320": 20500,
    "LOSSLESS": 22050,
}

# Cutoff detection tolerance (Hz)
CUTOFF_TOLERANCE = 500

# Cutoff detector parameters
ENERGY_EPSILON = 1e-10  # Minimum energy threshold to avoid division by zero
GRADIENT_WINDOW_HZ = 200  # Window size for gradient measurement (Hz)
GRADIENT_NORMALIZATION_DB = 30.0  # dB drop that corresponds to gradient=1.0

# Spectral analysis parameters - extended for lossless detection
SPECTRAL_PARAMS = {
    "min_freq": 16000,   # Paper's starting frequency (Hz)
    "max_freq": 22050,   # Extended to Nyquist for lossless detection (Hz)
    "num_bands": 150,    # 100 (paper) + 50 (ultrasonic extension)
    "fft_size": 8192,    # FFT window size
}

# SVM parameters from D'Alessandro & Shi paper (97% accuracy)
CLASSIFIER_PARAMS = {
    "kernel": "poly",
    "degree": 2,       # d=2 from paper - CRITICAL
    "gamma": 1,        # γ=1 from paper
    "C": 1,            # C=1 from paper
    "coef0": 1,        # Standard for polynomial kernel
    "probability": True,  # For confidence scores
    "cache_size": 1000,
    "class_weight": "balanced",
    "random_state": 42,
}

# Lossless container formats
LOSSLESS_CONTAINERS = {"flac", "wav", "alac", "ape", "wv", "aiff"}

# Lossy container formats
LOSSY_CONTAINERS = {"mp3", "aac", "ogg", "opus", "m4a", "wma"}

# Encoder signatures for detection
ENCODER_SIGNATURES = {
    "LAME": ["LAME", "Lavf"],
    "FhG": ["FhG"],
    "Xing": ["Xing"],
    "Fraunhofer": ["Fraunhofer"],
    "BladeEnc": ["BladeEnc"],
}

# Minimum requirements
MINIMUM_SAMPLE_RATE = 44100  # Need at least 44.1kHz for spectral analysis
MINIMUM_DURATION = 0.1  # At least 100ms of audio

# Warning thresholds
LOW_CONFIDENCE_THRESHOLD = 0.7  # Warn if confidence below this
BITRATE_MISMATCH_FACTOR = 1.5  # Warn if stated bitrate > detected * this factor
