#!/usr/bin/env python3
"""Validate new features by measuring separation on paired training data.

Computes per-feature separation metrics (Cohen's d style) between class pairs
using stem-matched files from the training data. Reports pass/fail per feature.

Usage:
    uv run python scripts/validate_features.py \
        --source-dir /mnt/bitrater/training_data/encoded \
        --pairs 20
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from bitrater.constants import FEATURE_NAMES
from bitrater.spectrum import SpectrumAnalyzer

# --- Configuration ---

# Indices of the 6 MDCT forensic features in the 193-element vector
NEW_FEATURE_START = 187  # mdct_zero_ratio_mean
NEW_FEATURE_END = 193  # exclusive (6 MDCT features at 187-192)
NEW_FEATURE_NAMES = FEATURE_NAMES[NEW_FEATURE_START:NEW_FEATURE_END]

# Class pairs to compare: (label_a, subdir_a, label_b, subdir_b)
# Problem pairs (original 3 — lowest accuracy class boundaries)
PROBLEM_PAIRS = [
    ("V0", "lossy/v0", "LOSSLESS", "lossless"),
    ("V2", "lossy/v2", "192", "lossy/192"),
    ("128", "lossy/128", "V2", "lossy/v2"),
]

# All adjacent pairs in quality order: 128 < V2 < 192 < V0 < 256 < 320 < LOSSLESS
ALL_PAIRS = [
    ("128", "lossy/128", "V2", "lossy/v2"),
    ("V2", "lossy/v2", "192", "lossy/192"),
    ("192", "lossy/192", "V0", "lossy/v0"),
    ("V0", "lossy/v0", "256", "lossy/256"),
    ("256", "lossy/256", "320", "lossy/320"),
    ("320", "lossy/320", "LOSSLESS", "lossless"),
    # Wide-gap pairs
    ("128", "lossy/128", "LOSSLESS", "lossless"),
    ("128", "lossy/128", "320", "lossy/320"),
    ("V0", "lossy/v0", "LOSSLESS", "lossless"),
]

# Thresholds
SEPARATION_THRESHOLD = 0.2
CONSISTENCY_THRESHOLD = 0.7


def find_paired_stems(dir_a: Path, dir_b: Path) -> list[str]:
    """Find file stems common to both directories."""
    stems_a = {p.stem for p in dir_a.iterdir() if p.is_file()}
    stems_b = {p.stem for p in dir_b.iterdir() if p.is_file()}
    return sorted(stems_a & stems_b)


def resolve_file(directory: Path, stem: str) -> Path | None:
    """Find the actual file for a stem in a directory (handles .mp3/.flac)."""
    for ext in (".mp3", ".flac", ".wav"):
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Try symlinks (lossless dir may have symlinks)
    for p in directory.iterdir():
        if p.stem == stem and p.is_file():
            return p
    return None


def extract_new_features(analyzer: SpectrumAnalyzer, file_path: Path) -> np.ndarray | None:
    """Extract only the 12 new features from a file."""
    sf = analyzer.analyze_file(str(file_path))
    if sf is None:
        return None
    vec = sf.as_vector()
    return vec[NEW_FEATURE_START:NEW_FEATURE_END]


def compute_separation(vals_a: np.ndarray, vals_b: np.ndarray):
    """Compute separation metric and consistency for paired values.

    Returns (separation, higher_class, consistency, n_valid) where:
    - separation = |mean_a - mean_b| / pooled_std  (Cohen's d style)
    - higher_class = 'A' or 'B' indicating which has higher mean
    - consistency = fraction of pairs where sign matches majority direction
    - n_valid = number of non-NaN pairs used
    """
    # Drop pairs where either value is NaN
    mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
    va, vb = vals_a[mask], vals_b[mask]
    n_valid = len(va)

    if n_valid < 3:
        return 0.0, "A", 0.0, n_valid

    mean_a, mean_b = np.mean(va), np.mean(vb)
    var_a = np.var(va, ddof=1)
    var_b = np.var(vb, ddof=1)
    pooled_std = np.sqrt((max(var_a, 0.0) + max(var_b, 0.0)) / 2)

    if pooled_std < 1e-10:
        return 0.0, "A", 0.0, n_valid

    separation = abs(mean_a - mean_b) / pooled_std
    higher_class = "A" if mean_a > mean_b else "B"

    # Consistency: fraction of pairs where direction matches majority
    diffs = va - vb
    if mean_a > mean_b:
        consistency = np.mean(diffs > 0)
    else:
        consistency = np.mean(diffs < 0)

    return separation, higher_class, consistency, n_valid


def validate_pair(
    analyzer: SpectrumAnalyzer,
    label_a: str,
    dir_a: Path,
    label_b: str,
    dir_b: Path,
    n_pairs: int,
    seed: int,
    n_threads: int,
) -> list[dict]:
    """Validate features for one class pair. Returns list of per-feature results."""
    # Find matched stems
    stems = find_paired_stems(dir_a, dir_b)
    if not stems:
        print(f"  ERROR: No matched stems between {dir_a} and {dir_b}")
        return []

    # Sample
    rng = random.Random(seed)
    sampled = rng.sample(stems, min(n_pairs, len(stems)))
    actual_pairs = len(sampled)
    print(f"  Found {len(stems)} matched stems, sampling {actual_pairs}")

    # Resolve file paths
    files_a = []
    files_b = []
    used_stems = []
    for stem in sampled:
        fa = resolve_file(dir_a, stem)
        fb = resolve_file(dir_b, stem)
        if fa and fb:
            files_a.append(fa)
            files_b.append(fb)
            used_stems.append(stem)

    if not files_a:
        print("  ERROR: Could not resolve any paired files")
        return []

    # Extract features in parallel
    all_files = files_a + files_b

    def _extract(f):
        return extract_new_features(analyzer, f)

    results = Parallel(n_jobs=n_threads, backend="threading")(
        delayed(_extract)(f) for f in all_files
    )

    # Split results back into A and B
    n = len(files_a)
    feats_a_list = results[:n]
    feats_b_list = results[n:]

    # Filter out failures (keep only pairs where both succeeded)
    valid_a, valid_b = [], []
    for fa, fb in zip(feats_a_list, feats_b_list, strict=True):
        if fa is not None and fb is not None:
            valid_a.append(fa)
            valid_b.append(fb)

    if len(valid_a) < 3:
        print(f"  ERROR: Only {len(valid_a)} valid pairs (need at least 3)")
        return []

    print(f"  Successfully extracted {len(valid_a)} pairs")

    feats_a = np.array(valid_a)  # shape: (n_valid, 12)
    feats_b = np.array(valid_b)

    # Compute per-feature metrics
    n_total = len(valid_a)
    pair_results = []
    for i, name in enumerate(NEW_FEATURE_NAMES):
        sep, higher, cons, n_valid = compute_separation(feats_a[:, i], feats_b[:, i])
        higher_label = label_a if higher == "A" else label_b
        passed = sep > SEPARATION_THRESHOLD and cons > CONSISTENCY_THRESHOLD and n_valid >= 3
        pair_results.append(
            {
                "feature": name,
                "separation": sep,
                "higher": higher_label,
                "consistency": cons,
                "passed": passed,
                "n_valid": n_valid,
                "n_total": n_total,
            }
        )

    return pair_results


def print_pair_table(label_a: str, label_b: str, n_pairs: int, results: list[dict]):
    """Print results table for one class pair."""
    print()
    print("=" * 70)
    print(f"{label_a} <-> {label_b} ({n_pairs} paired tracks)")
    print("=" * 70)
    print(f"{'Feature':<35} {'Sep.':>6} {'Higher':<10} {'Consist.':>9}  {'N':>4}  {'Result'}")
    print("-" * 74)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        n_note = f"{r['n_valid']}" if r["n_valid"] < r["n_total"] else ""
        print(
            f"{r['feature']:<35} {r['separation']:>6.2f} {r['higher']:<10} "
            f"{r['consistency']:>8.1%}  {n_note:>4}  {status}"
        )
    n_pass = sum(1 for r in results if r["passed"])
    print("-" * 70)
    print(f"{n_pass}/{len(results)} features PASS for this pair")


def print_summary(all_results: dict[str, list[dict]]):
    """Print overall summary table."""
    pair_labels = list(all_results.keys())
    n_pairs = len(pair_labels)

    # Abbreviate pair labels for compact columns
    abbrevs = []
    for pl in pair_labels:
        a, b = pl.split("<->")
        short = f"{a[:3]}-{b[:3]}"
        abbrevs.append(short)

    col_w = max(7, max(len(a) for a in abbrevs) + 1)
    name_w = 35
    total_w = name_w + col_w * n_pairs + 10 + 8  # verdict + pass count

    print()
    print("=" * total_w)
    print("OVERALL FEATURE VALIDATION SUMMARY")
    print("=" * total_w)

    header = f"{'Feature':<{name_w}}"
    for ab in abbrevs:
        header += f" {ab:>{col_w}}"
    header += f"  {'#':>3}  Verdict"
    print(header)
    print("-" * total_w)

    for i, name in enumerate(NEW_FEATURE_NAMES):
        row = f"{name:<{name_w}}"
        pass_count = 0
        for pl in pair_labels:
            results = all_results[pl]
            if i < len(results):
                if results[i]["passed"]:
                    status = "PASS"
                    pass_count += 1
                else:
                    status = "fail"
            else:
                status = "n/a"
            row += f" {status:>{col_w}}"

        if pass_count == n_pairs:
            verdict = "KEEP"
        elif pass_count == 0:
            verdict = "DROP?"
        else:
            verdict = "REVIEW"
        row += f"  {pass_count:>3}  {verdict}"
        print(row)

    print("-" * total_w)
    print(f"KEEP: passes all {n_pairs} pairs | REVIEW: passes some | DROP?: passes none")


def main():
    parser = argparse.ArgumentParser(
        description="Validate new features on paired training data"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Training data root (e.g., /mnt/bitrater/training_data/encoded)",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=20,
        help="Number of paired tracks per class pair (default: 20)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Parallel workers for feature extraction (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Test all adjacent + wide-gap pairs (9 total) instead of 3 problem pairs",
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"ERROR: Source directory not found: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    pairs = ALL_PAIRS if args.all_pairs else PROBLEM_PAIRS
    mode = "all pairs (adjacent + wide-gap)" if args.all_pairs else "problem pairs only"

    print(f"Feature validation: {len(NEW_FEATURE_NAMES)} new features")
    print(f"Features: {', '.join(NEW_FEATURE_NAMES)}")
    print(f"Source: {args.source_dir}")
    print(f"Mode: {mode} ({len(pairs)} comparisons)")
    print(f"Pairs per comparison: {args.pairs}, Seed: {args.seed}")
    print()

    analyzer = SpectrumAnalyzer()
    all_results: dict[str, list[dict]] = {}

    for label_a, subdir_a, label_b, subdir_b in pairs:
        dir_a = args.source_dir / subdir_a
        dir_b = args.source_dir / subdir_b

        pair_key = f"{label_a}<->{label_b}"
        print(f"Processing {pair_key}...")

        if not dir_a.is_dir():
            print(f"  SKIP: Directory not found: {dir_a}")
            continue
        if not dir_b.is_dir():
            print(f"  SKIP: Directory not found: {dir_b}")
            continue

        results = validate_pair(
            analyzer, label_a, dir_a, label_b, dir_b,
            args.pairs, args.seed, args.threads,
        )

        if results:
            all_results[pair_key] = results
            print_pair_table(label_a, label_b, args.pairs, results)

    if all_results:
        print_summary(all_results)
    else:
        print("\nNo pairs could be validated. Check --source-dir path.")
        sys.exit(1)


if __name__ == "__main__":
    main()
