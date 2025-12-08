"""Deep learning inference pipeline for bitrate classification.

Uses ONNX Runtime for inference — no PyTorch dependency required.
Loads audio once, shares STFT across all feature extractors
(dual-band CNN, global modulation, SVM features).
"""

import logging
from pathlib import Path

import numpy as np
import librosa
import onnxruntime as ort
from scipy.fft import dctn
from scipy.special import softmax

from .spectrum import SpectrumAnalyzer

logger = logging.getLogger("beets.bitrater")

# --- Constants (inlined from training config) ---

CLASS_NAMES = ["128", "V2", "192", "V0", "256", "320", "LOSSLESS"]
NUM_CLASSES = len(CLASS_NAMES)

# Spectrogram parameters
_SAMPLE_RATE = 44100
_HOP_LENGTH = 512
_WINDOW_SEC = 2.0
_HOP_SEC = 1.0

# High-frequency band parameters (dual-band filterbank)
_HF_N_FFT = 2048
_HF_FMIN = 16000
_HF_FMAX = 22050
_HF_BINS = 64

# Sequence model parameters
_SEQ_LEN = 48

# Global modulation DCT parameters
_DCT_ROWS = 5
_DCT_COLS = 6
_N_GLOBAL_FEATURES = _DCT_ROWS * _DCT_COLS  # 30

# SVM feature count
_N_SVM_FEATURES = 181

# Default model directory (inside package)
_MODELS_DIR = Path(__file__).parent / "models"


class BitrateInference:
    """Optimized single-file inference pipeline using ONNX Runtime.

    Loads audio once, shares STFT across all feature extractors,
    and only processes the CNN windows needed for sequence coverage.
    """

    def __init__(self, cnn_session, seq_session):
        self.cnn_session = cnn_session
        self.seq_session = seq_session
        self.spectrum_analyzer = SpectrumAnalyzer()

    def predict(self, audio_path):
        """Classify a single audio file.

        Returns: (class_name, confidence, probabilities_dict) or (None, 0.0, {}) on failure.
        """
        # 1. Load audio once
        audio, _ = librosa.load(audio_path, sr=_SAMPLE_RATE, mono=True)

        # 2. Compute shared STFT
        S_complex = librosa.stft(audio, n_fft=_HF_N_FFT, hop_length=_HOP_LENGTH, center=True)
        S_power = np.abs(S_complex) ** 2
        freqs = librosa.fft_frequencies(sr=_SAMPLE_RATE, n_fft=_HF_N_FFT)

        # 3. Extract dual-band spectrograms for CNN (windowed)
        window_samples = int(_WINDOW_SEC * _SAMPLE_RATE)
        hop_samples = int(_HOP_SEC * _SAMPLE_RATE)
        n_stft_frames_per_window = 1 + window_samples // _HOP_LENGTH

        total_windows = max(0, (len(audio) - window_samples) // hop_samples + 1)
        if total_windows < _SEQ_LEN:
            needed_windows = list(range(total_windows))
        else:
            stride = _SEQ_LEN // 2
            seq_starts = list(range(0, total_windows - _SEQ_LEN + 1, stride))
            needed = set()
            for s in seq_starts:
                for w in range(s, s + _SEQ_LEN):
                    needed.add(w)
            needed_windows = sorted(needed)

        # Build filterbanks
        mel_basis = librosa.filters.mel(
            sr=_SAMPLE_RATE, n_fft=_HF_N_FFT, n_mels=64, fmin=0, fmax=_HF_FMIN,
        )
        hf_start = np.searchsorted(freqs, _HF_FMIN)
        hf_end = np.searchsorted(freqs, _HF_FMAX)

        n_hf_bins = hf_end - hf_start
        hf_edges = (
            np.linspace(0, n_hf_bins, _HF_BINS + 1, dtype=int)
            if n_hf_bins >= _HF_BINS
            else None
        )

        specs = []
        for win_idx in needed_windows:
            audio_start = win_idx * hop_samples
            stft_start = audio_start // _HOP_LENGTH
            stft_end = min(stft_start + n_stft_frames_per_window, S_power.shape[1])

            S_win = S_power[:, stft_start:stft_end]

            # Band 1: Mel (0-16 kHz, 64 bins)
            mel_part = mel_basis @ S_win

            # Band 2: Linear (16-22 kHz, 64 bins)
            hf_raw = S_win[hf_start:hf_end, :]
            if hf_edges is not None:
                hf_part = np.array([
                    hf_raw[hf_edges[i]:hf_edges[i + 1]].mean(axis=0)
                    for i in range(_HF_BINS)
                ])
            else:
                hf_part = np.zeros((_HF_BINS, S_win.shape[1]), dtype=np.float32)
                hf_part[:n_hf_bins] = hf_raw

            # Pad to consistent time frames
            target_t = n_stft_frames_per_window
            if mel_part.shape[1] < target_t:
                mel_part = np.pad(mel_part, ((0, 0), (0, target_t - mel_part.shape[1])))
                hf_part = np.pad(hf_part, ((0, 0), (0, target_t - hf_part.shape[1])))
            elif mel_part.shape[1] > target_t:
                mel_part = mel_part[:, :target_t]
                hf_part = hf_part[:, :target_t]

            combined = np.vstack([mel_part, hf_part])
            combined_db = librosa.power_to_db(combined, ref=np.max)
            spec_min = combined_db.min()
            spec_max = combined_db.max()
            normalized = (combined_db - spec_min) / (spec_max - spec_min + 1e-8)
            specs.append(normalized[np.newaxis, ...].astype(np.float32))

        if not specs:
            return None, 0.0, {}

        # 4. CNN forward pass via ONNX Runtime
        specs_arr = np.stack(specs)
        cnn_features = self.cnn_session.run(None, {"spectrogram": specs_arr})[0]

        win_to_feat = {w: i for i, w in enumerate(needed_windows)}

        # 5. Global modulation features from shared STFT
        mel_full = mel_basis @ S_power
        hf_full_raw = S_power[hf_start:hf_end, :]
        if hf_edges is not None:
            hf_full = np.array([
                hf_full_raw[hf_edges[i]:hf_edges[i + 1]].mean(axis=0)
                for i in range(_HF_BINS)
            ])
        else:
            hf_full = np.zeros((_HF_BINS, S_power.shape[1]), dtype=np.float32)
            hf_full[:n_hf_bins] = hf_full_raw

        combined_full = np.vstack([mel_full, hf_full])
        log_spec = np.log(combined_full + 1e-10)

        target_frames = 250
        if log_spec.shape[1] < target_frames:
            padded = np.zeros((128, target_frames), dtype=np.float32)
            padded[:, :log_spec.shape[1]] = log_spec
            log_spec = padded
        elif log_spec.shape[1] > target_frames:
            start = (log_spec.shape[1] - target_frames) // 2
            log_spec = log_spec[:, start:start + target_frames]

        dct_coeffs = dctn(log_spec, norm="ortho")
        gm_feats = dct_coeffs[:_DCT_ROWS, :_DCT_COLS].flatten().astype(np.float32)
        l1_norm = np.abs(gm_feats).sum()
        if l1_norm > 0:
            gm_feats = gm_feats / l1_norm

        # 6. SVM spectral features
        svm_result = self.spectrum_analyzer.analyze_file(audio_path, is_vbr=0.0)
        if svm_result is not None:
            svm_feats = svm_result.as_vector().astype(np.float32)
        else:
            svm_feats = np.zeros(_N_SVM_FEATURES, dtype=np.float32)

        # 7. Combine auxiliary features
        aux = np.concatenate([svm_feats, gm_feats])[np.newaxis, ...]  # (1, 211)

        # 8. Sequence model inference with sliding window via ONNX Runtime
        stride = _SEQ_LEN // 2
        all_probs = []

        for seq_start in range(0, total_windows - _SEQ_LEN + 1, stride):
            seq_feats = []
            for w in range(seq_start, seq_start + _SEQ_LEN):
                if w in win_to_feat:
                    seq_feats.append(cnn_features[win_to_feat[w]])
                else:
                    seq_feats.append(np.zeros(cnn_features.shape[1], dtype=np.float32))

            seq_input = np.stack(seq_feats)[np.newaxis, ...]  # (1, 48, 128)
            logits = self.seq_session.run(
                None, {"cnn_features": seq_input, "aux_features": aux}
            )[0]  # (1, 7)
            all_probs.append(softmax(logits[0]))

        if not all_probs:
            return None, 0.0, {}

        # 9. Aggregate predictions
        avg_probs = np.mean(all_probs, axis=0)
        pred_idx = int(avg_probs.argmax())
        confidence = float(avg_probs[pred_idx])
        class_name = CLASS_NAMES[pred_idx]

        probs_dict = {name: float(avg_probs[i]) for i, name in enumerate(CLASS_NAMES)}

        return class_name, confidence, probs_dict


def load_inference_pipeline(stage1_path=None, stage2_path=None):
    """Load ONNX models and create inference pipeline.

    Args:
        stage1_path: Path to stage 1 CNN ONNX model. Defaults to bundled model.
        stage2_path: Path to stage 2 sequence ONNX model. Defaults to bundled model.

    Returns:
        BitrateInference instance, or None if model files not found.
    """
    if stage1_path is None:
        stage1_path = _MODELS_DIR / "stage1_cnn.onnx"
    if stage2_path is None:
        stage2_path = _MODELS_DIR / "stage2_seq.onnx"

    stage1_path = Path(stage1_path)
    stage2_path = Path(stage2_path)

    if not stage1_path.exists() or not stage2_path.exists():
        return None

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1

    cnn_session = ort.InferenceSession(str(stage1_path), opts)
    seq_session = ort.InferenceSession(str(stage2_path), opts)

    logger.info("Loaded deep learning model (ONNX Runtime)")
    return BitrateInference(cnn_session, seq_session)
