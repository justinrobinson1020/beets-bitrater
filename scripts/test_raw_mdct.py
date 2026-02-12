#!/usr/bin/env python3
"""Test raw MP3 MDCT extraction methods.

Compares MDCT features computed from decoded PCM (current approach) against
features computed with known frame alignment from MP3 header parsing.

Also extracts bitstream-level features (frame sizes, bitrate counts) that are
unavailable from decoded PCM.

Usage:
    uv run python scripts/test_raw_mdct.py \
        --source-dir /mnt/bitrater/training_data/encoded \
        --files-per-class 5
"""

import argparse
import struct
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy import stats

# ── MP3 frame header parsing ────────────────────────────────────────────────

# MPEG1 Layer 3 bitrate table (index 0 = free, 15 = bad)
MPEG1_L3_BITRATES = [
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, -1
]

# MPEG1 sample rate table
MPEG1_SAMPLERATES = [44100, 48000, 32000, -1]

# Samples per frame for MPEG1 Layer 3
SAMPLES_PER_FRAME = 1152


def parse_mp3_frames(filepath: Path) -> dict:
    """Parse MP3 frame headers to extract frame boundaries and bitstream stats.

    Returns dict with:
        frame_sizes: list of frame sizes in bytes
        frame_bitrates: list of per-frame bitrates in kbps
        frame_sample_offsets: list of sample offsets for each frame start
        encoder_delay: encoder delay from LAME/Xing header (0 if not found)
        n_frames: total frame count
    """
    data = filepath.read_bytes()
    pos = 0
    max_pos = len(data) - 4

    # Skip ID3v2 tag if present
    if data[:3] == b"ID3":
        id3_size = (
            (data[6] & 0x7F) << 21
            | (data[7] & 0x7F) << 14
            | (data[8] & 0x7F) << 7
            | (data[9] & 0x7F)
        )
        pos = 10 + id3_size

    frame_sizes = []
    frame_bitrates = []
    frame_sample_offsets = []
    encoder_delay = 0
    sample_offset = 0
    first_frame = True

    while pos < max_pos:
        # Find sync word: 11 set bits (0xFFE0 mask)
        header_bytes = data[pos : pos + 4]
        if len(header_bytes) < 4:
            break

        header = struct.unpack(">I", header_bytes)[0]

        # Check sync (bits 31-21 = 0x7FF)
        if (header >> 21) != 0x7FF:
            pos += 1
            continue

        # Parse header fields
        version = (header >> 19) & 0x03  # 11=MPEG1, 10=MPEG2, 00=MPEG2.5
        layer = (header >> 17) & 0x03  # 01=Layer3
        br_idx = (header >> 12) & 0x0F
        sr_idx = (header >> 10) & 0x03
        padding = (header >> 9) & 0x01

        # Only handle MPEG1 Layer 3
        if version != 3 or layer != 1:
            pos += 1
            continue

        if br_idx == 0 or br_idx == 15 or sr_idx == 3:
            pos += 1
            continue

        bitrate = MPEG1_L3_BITRATES[br_idx]
        samplerate = MPEG1_SAMPLERATES[sr_idx]

        # Frame size = 144 * bitrate / samplerate + padding
        frame_size = (144 * bitrate * 1000) // samplerate + padding

        if frame_size < 24:  # Sanity check
            pos += 1
            continue

        # Check for Xing/LAME header in first frame
        if first_frame:
            first_frame = False
            # Side information is 32 bytes for MPEG1 stereo, 17 for mono
            # Xing header typically at offset 36 (stereo) or 21 (mono)
            for xing_offset in [36, 21]:
                xing_pos = pos + 4 + xing_offset
                if xing_pos + 4 <= len(data):
                    tag = data[xing_pos : xing_pos + 4]
                    if tag in (b"Xing", b"Info"):
                        # Look for LAME tag (at +120 from Xing)
                        lame_pos = xing_pos + 120
                        if lame_pos + 9 <= len(data):
                            lame_tag = data[lame_pos : lame_pos + 4]
                            if lame_tag == b"LAME":
                                # Encoder delay is at LAME+21 (12 bits)
                                delay_pos = lame_pos + 21
                                if delay_pos + 3 <= len(data):
                                    delay_bytes = data[delay_pos : delay_pos + 3]
                                    encoder_delay = (
                                        (delay_bytes[0] << 4)
                                        | (delay_bytes[1] >> 4)
                                    )
                        break

        frame_sizes.append(frame_size)
        frame_bitrates.append(bitrate)
        frame_sample_offsets.append(sample_offset)
        sample_offset += SAMPLES_PER_FRAME

        pos += frame_size

    return {
        "frame_sizes": frame_sizes,
        "frame_bitrates": frame_bitrates,
        "frame_sample_offsets": frame_sample_offsets,
        "encoder_delay": encoder_delay,
        "n_frames": len(frame_sizes),
    }


# ── Bitstream features (from raw MP3 headers) ───────────────────────────────


def extract_bitstream_features(frame_info: dict) -> dict:
    """Extract features from raw MP3 frame headers (unavailable from decoded PCM).

    Returns dict of feature_name -> value.
    """
    sizes = np.array(frame_info["frame_sizes"], dtype=np.float64)
    bitrates = np.array(frame_info["frame_bitrates"], dtype=np.float64)

    if len(sizes) < 2:
        return {
            "frame_size_var": 0.0,
            "frame_size_cv": 0.0,
            "unique_bitrate_count": 0,
            "frame_size_entropy": 0.0,
            "bitrate_range": 0.0,
        }

    # Frame size variance: CBR ≈ 0 (only padding variation), VBR >> 0
    frame_size_var = float(np.var(sizes))

    # Coefficient of variation (normalized variance)
    frame_size_cv = float(np.std(sizes) / (np.mean(sizes) + 1e-10))

    # Unique bitrate count: CBR = 1, VBR = many
    unique_bitrate_count = len(np.unique(bitrates))

    # Frame size entropy
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    probs = counts / counts.sum()
    frame_size_entropy = float(stats.entropy(probs, base=2))

    # Bitrate range
    bitrate_range = float(np.max(bitrates) - np.min(bitrates))

    return {
        "frame_size_var": frame_size_var,
        "frame_size_cv": frame_size_cv,
        "unique_bitrate_count": unique_bitrate_count,
        "frame_size_entropy": frame_size_entropy,
        "bitrate_range": bitrate_range,
    }


# ── MDCT computation at a given offset ───────────────────────────────────────

N_WINDOW = 1152
HALF_N = N_WINDOW // 2  # 576


def compute_mdct_stats(y: np.ndarray, sr: int, offset: int) -> dict:
    """Compute MDCT coefficient statistics at a given sample offset.

    Returns the 7 "flat" features that fail on misaligned decoded PCM.
    """
    window = np.sin(np.pi * (np.arange(N_WINDOW) + 0.5) / N_WINDOW)

    max_samples = min(len(y), sr * 10)
    audio = np.ascontiguousarray(y[:max_samples], dtype=np.float64)

    seg = audio[offset:]
    if len(seg) < N_WINDOW * 4:
        return _empty_stats()

    frames_T = librosa.util.frame(seg, frame_length=N_WINDOW, hop_length=HALF_N)
    n_frames = min(frames_T.shape[1], 400)
    if n_frames < 4:
        return _empty_stats()

    frames = frames_T[:, :n_frames].T
    windowed = frames * window

    # MDCT basis
    n_idx = np.arange(N_WINDOW)[:, np.newaxis]
    k_idx = np.arange(HALF_N)[np.newaxis, :]
    basis = np.cos(np.pi / HALF_N * (n_idx + 0.5 + HALF_N / 2) * (k_idx + 0.5))
    coeffs = windowed @ basis

    # Threshold
    median_abs = np.median(np.abs(coeffs))
    threshold = max(median_abs * 0.01, 1e-10)
    zero_mask = np.abs(coeffs) < threshold

    # Zero ratio (frame-level) — for alignment energy comparison
    zero_ratio_per_frame = np.mean(zero_mask, axis=1)

    # Alignment score proxy: windowed energy variance across test offsets
    frame_energies = np.mean(windowed**2, axis=1)
    energy_db = 10 * np.log10(np.mean(frame_energies) + 1e-10)

    # Coefficient distribution features
    flat = coeffs.flatten()
    nonzero = flat[np.abs(flat) >= threshold]
    if len(nonzero) > 100:
        coeff_kurtosis = float(stats.kurtosis(nonzero, fisher=True))
        coeff_skewness = float(stats.skew(nonzero))
        hist_vals, _ = np.histogram(nonzero, bins=50, density=True)
        hist_vals = hist_vals[hist_vals > 0]
        coeff_entropy = float(stats.entropy(hist_vals))
    else:
        coeff_kurtosis = 0.0
        coeff_skewness = 0.0
        coeff_entropy = 0.0

    # Inter-frame correlation
    n_corr = min(n_frames - 1, 200)
    frame_corrs = []
    for i in range(n_corr):
        c = np.corrcoef(np.abs(coeffs[i]), np.abs(coeffs[i + 1]))[0, 1]
        if not np.isnan(c):
            frame_corrs.append(c)

    interframe_corr_mean = float(np.mean(frame_corrs)) if frame_corrs else 0.0
    interframe_corr_var = float(np.var(frame_corrs)) if frame_corrs else 0.0

    return {
        "energy_db": energy_db,
        "zero_ratio_mean": float(np.mean(zero_ratio_per_frame)),
        "coeff_kurtosis": coeff_kurtosis,
        "coeff_skewness": coeff_skewness,
        "coeff_entropy": coeff_entropy,
        "interframe_corr_mean": interframe_corr_mean,
        "interframe_corr_var": interframe_corr_var,
    }


def _empty_stats():
    return {
        "energy_db": 0.0,
        "zero_ratio_mean": 0.0,
        "coeff_kurtosis": 0.0,
        "coeff_skewness": 0.0,
        "coeff_entropy": 0.0,
        "interframe_corr_mean": 0.0,
        "interframe_corr_var": 0.0,
    }


# ── Test A: Aligned vs misaligned MDCT ───────────────────────────────────────


def test_alignment(filepath: Path, frame_info: dict) -> dict:
    """Compare MDCT stats at frame-aligned offset vs random offsets."""
    y, sr = librosa.load(str(filepath), sr=44100, mono=True)

    # Compute aligned offset from encoder delay
    delay = frame_info["encoder_delay"]
    aligned_offset = delay % HALF_N if delay > 0 else 0

    # Stats at aligned offset
    aligned_stats = compute_mdct_stats(y, sr, aligned_offset)

    # Stats at 10 random offsets (misaligned)
    rng = np.random.default_rng(42)
    random_offsets = rng.integers(0, HALF_N, size=10)
    # Exclude the aligned offset
    random_offsets = [o for o in random_offsets if abs(o - aligned_offset) > 10][:10]

    misaligned_stats_list = []
    for offset in random_offsets:
        s = compute_mdct_stats(y, sr, int(offset))
        misaligned_stats_list.append(s)

    # Average misaligned stats
    avg_misaligned = {}
    for key in aligned_stats:
        vals = [s[key] for s in misaligned_stats_list]
        avg_misaligned[key] = float(np.mean(vals))

    return {
        "aligned": aligned_stats,
        "misaligned_avg": avg_misaligned,
        "encoder_delay": delay,
        "aligned_offset": aligned_offset,
    }


# ── Cohens d effect size ────────────────────────────────────────────────────


def cohens_d(group_a: list, group_b: list) -> float:
    """Compute Cohen's d effect size between two groups."""
    a, b = np.array(group_a), np.array(group_b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


# ── Main ─────────────────────────────────────────────────────────────────────

CLASS_DIRS = {
    "128": "lossy/128",
    "V2": "lossy/v2",
    "192": "lossy/192",
    "V0": "lossy/v0",
    "320": "lossy/320",
    "LOSSLESS": "lossless",
}


def collect_files(source_dir: Path, files_per_class: int, seed: int = 42) -> dict:
    """Collect sample files from each class."""
    rng = np.random.default_rng(seed)
    result = {}
    for label, subdir in CLASS_DIRS.items():
        class_dir = source_dir / subdir
        if not class_dir.exists():
            print(f"  SKIP {label}: {class_dir} not found")
            continue
        if label == "LOSSLESS":
            files = sorted(class_dir.glob("*.flac"))
        else:
            files = sorted(class_dir.glob("*.mp3"))
        if not files:
            print(f"  SKIP {label}: no files found")
            continue
        chosen = rng.choice(files, size=min(files_per_class, len(files)), replace=False)
        result[label] = list(chosen)
    return result


def main():
    parser = argparse.ArgumentParser(description="Test raw MP3 MDCT extraction")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/mnt/bitrater/training_data/encoded"),
    )
    parser.add_argument("--files-per-class", type=int, default=5)
    args = parser.parse_args()

    print("=" * 78)
    print("Test Raw MP3 MDCT Extraction Methods")
    print("=" * 78)

    # Collect files
    print(f"\nCollecting {args.files_per_class} files per class...")
    files_by_class = collect_files(args.source_dir, args.files_per_class)

    if not files_by_class:
        print("ERROR: No files found")
        sys.exit(1)

    for label, files in files_by_class.items():
        print(f"  {label}: {len(files)} files")

    # ── Test A: Aligned vs misaligned MDCT ──────────────────────────────
    print("\n" + "=" * 78)
    print("TEST A: Aligned vs Misaligned MDCT (MP3 files only)")
    print("=" * 78)
    print("\nDo the 7 flat features differ when computed at the correct frame alignment?")

    alignment_results = {}
    for label, files in files_by_class.items():
        if label == "LOSSLESS":
            continue  # No MP3 frames to parse
        print(f"\n  Processing {label}...")
        class_results = []
        for f in files:
            frame_info = parse_mp3_frames(f)
            if frame_info["n_frames"] < 10:
                print(f"    SKIP {f.name}: only {frame_info['n_frames']} frames parsed")
                continue
            result = test_alignment(f, frame_info)
            class_results.append(result)
            print(
                f"    {f.name}: delay={result['encoder_delay']}, "
                f"offset={result['aligned_offset']}, "
                f"frames={frame_info['n_frames']}"
            )
        alignment_results[label] = class_results

    # Report alignment differences
    feat_keys = [
        "coeff_kurtosis",
        "coeff_skewness",
        "coeff_entropy",
        "interframe_corr_mean",
        "interframe_corr_var",
        "energy_db",
        "zero_ratio_mean",
    ]

    print(f"\n{'Feature':<25} {'Aligned':>10} {'Misaligned':>10} {'Diff%':>8}")
    print("-" * 55)
    for feat in feat_keys:
        all_aligned = []
        all_misaligned = []
        for label, results in alignment_results.items():
            for r in results:
                all_aligned.append(r["aligned"][feat])
                all_misaligned.append(r["misaligned_avg"][feat])

        mean_a = np.mean(all_aligned) if all_aligned else 0
        mean_m = np.mean(all_misaligned) if all_misaligned else 0
        denom = abs(mean_m) if abs(mean_m) > 1e-10 else 1e-10
        diff_pct = 100 * (mean_a - mean_m) / denom
        print(f"  {feat:<23} {mean_a:>10.4f} {mean_m:>10.4f} {diff_pct:>+7.1f}%")

    # Per-class aligned MDCT stats
    print(f"\n{'Class':<8}", end="")
    for feat in feat_keys:
        print(f" {feat[:12]:>12}", end="")
    print()
    print("-" * (8 + 12 * len(feat_keys) + len(feat_keys)))
    for label, results in alignment_results.items():
        print(f"  {label:<6}", end="")
        for feat in feat_keys:
            vals = [r["aligned"][feat] for r in results]
            print(f" {np.mean(vals):>12.4f}", end="")
        print()

    # ── Test B: Bitstream features ──────────────────────────────────────
    print("\n" + "=" * 78)
    print("TEST B: Bitstream Features (from raw MP3 frame headers)")
    print("=" * 78)
    print("\nThese features are ONLY available from raw MP3, not decoded PCM.")

    bitstream_by_class = {}
    for label, files in files_by_class.items():
        if label == "LOSSLESS":
            continue
        feats_list = []
        for f in files:
            frame_info = parse_mp3_frames(f)
            if frame_info["n_frames"] < 10:
                continue
            feats = extract_bitstream_features(frame_info)
            feats_list.append(feats)
        bitstream_by_class[label] = feats_list

    bs_feat_keys = [
        "frame_size_var",
        "frame_size_cv",
        "unique_bitrate_count",
        "frame_size_entropy",
        "bitrate_range",
    ]

    # Per-class means
    print(f"\n{'Class':<8}", end="")
    for feat in bs_feat_keys:
        print(f" {feat[:14]:>14}", end="")
    print()
    print("-" * (8 + 14 * len(bs_feat_keys) + len(bs_feat_keys)))
    for label in CLASS_DIRS:
        if label == "LOSSLESS" or label not in bitstream_by_class:
            continue
        feats_list = bitstream_by_class[label]
        if not feats_list:
            continue
        print(f"  {label:<6}", end="")
        for feat in bs_feat_keys:
            vals = [f[feat] for f in feats_list]
            print(f" {np.mean(vals):>14.4f}", end="")
        print()

    # Cohen's d for key class pairs
    print("\nCohen's d separation (|d| > 0.8 = large effect):")
    pairs = [
        ("V2", "192", "VBR vs CBR neighbor"),
        ("V0", "320", "VBR vs CBR neighbor"),
        ("128", "320", "Low vs high CBR"),
        ("V2", "V0", "VBR presets"),
    ]
    print(f"\n  {'Pair':<22}", end="")
    for feat in bs_feat_keys:
        print(f" {feat[:14]:>14}", end="")
    print()
    print("  " + "-" * (22 + 14 * len(bs_feat_keys) + len(bs_feat_keys)))
    for la, lb, desc in pairs:
        if la not in bitstream_by_class or lb not in bitstream_by_class:
            continue
        print(f"  {la} vs {lb:<14}", end="")
        for feat in bs_feat_keys:
            va = [f[feat] for f in bitstream_by_class[la]]
            vb = [f[feat] for f in bitstream_by_class[lb]]
            d = cohens_d(va, vb)
            marker = "***" if abs(d) > 0.8 else "  *" if abs(d) > 0.5 else "   "
            print(f" {d:>10.2f}{marker}", end="")
        print()

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    # Check if alignment matters
    print("\nTest A (Aligned MDCT):")
    for feat in feat_keys:
        all_aligned = []
        all_misaligned = []
        for results in alignment_results.values():
            for r in results:
                all_aligned.append(r["aligned"][feat])
                all_misaligned.append(r["misaligned_avg"][feat])
        if all_aligned and all_misaligned:
            d = cohens_d(all_aligned, all_misaligned)
            verdict = "ALIGNMENT MATTERS" if abs(d) > 0.5 else "no alignment effect"
            print(f"  {feat:<25} d={d:>+.3f}  -> {verdict}")

    print("\nTest B (Bitstream features):")
    # Check if any bitstream feature separates VBR from CBR
    cbr_labels = ["128", "192", "320"]
    vbr_labels = ["V2", "V0"]
    for feat in bs_feat_keys:
        cbr_vals = []
        vbr_vals = []
        for la in cbr_labels:
            if la in bitstream_by_class:
                cbr_vals.extend([f[feat] for f in bitstream_by_class[la]])
        for la in vbr_labels:
            if la in bitstream_by_class:
                vbr_vals.extend([f[feat] for f in bitstream_by_class[la]])
        if cbr_vals and vbr_vals:
            d = cohens_d(vbr_vals, cbr_vals)
            verdict = "DISCRIMINATIVE" if abs(d) > 0.8 else "weak" if abs(d) > 0.5 else "not useful"
            print(f"  {feat:<25} CBR-vs-VBR d={d:>+.3f}  -> {verdict}")

    print()


if __name__ == "__main__":
    main()
