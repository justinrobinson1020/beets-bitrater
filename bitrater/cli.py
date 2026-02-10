"""Standalone CLI for bitrater audio quality analysis."""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("bitrater")


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze audio files and print results."""
    from bitrater.analyzer import AudioQualityAnalyzer

    analyzer = AudioQualityAnalyzer()

    if args.model:
        analyzer.load_model(Path(args.model))
        logger.info(f"Loaded model from {args.model}")

    target = Path(args.target)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(
            p
            for p in target.rglob("*")
            if p.suffix.lower() in {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac"}
        )
    else:
        logger.error(f"Not a file or directory: {target}")
        sys.exit(1)

    if not files:
        logger.error(f"No audio files found in {target}")
        sys.exit(1)

    logger.info(f"Analyzing {len(files)} file(s)...")

    for filepath in files:
        result = analyzer.analyze_file(str(filepath))
        if result is None:
            logger.warning(f"  SKIP {filepath.name} (analysis failed)")
            continue

        status = "TRANSCODE" if result.is_transcode else "OK"
        print(
            f"[{status}] {filepath.name}: "
            f"{result.original_format} {result.original_bitrate}kbps "
            f"(confidence: {result.confidence:.0%})"
        )
        if args.verbose and result.warnings:
            for w in result.warnings:
                print(f"  warn: {w}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train classifier from a directory of labeled audio."""
    from bitrater.analyzer import AudioQualityAnalyzer

    analyzer = AudioQualityAnalyzer()
    source_dir = Path(args.source_dir)
    save_path = Path(args.save_model) if args.save_model else None

    logger.info(f"Training from {source_dir}...")
    stats = analyzer.train_from_directory(
        source_dir, save_path=save_path, num_workers=args.threads
    )

    logger.info(
        f"Training complete: {stats['successful']}/{stats['total_files']} files processed"
    )
    if save_path:
        logger.info(f"Model saved to {save_path}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate classifier accuracy with train/test split."""
    from bitrater.analyzer import AudioQualityAnalyzer

    analyzer = AudioQualityAnalyzer()
    source_dir = Path(args.source_dir)

    logger.info(f"Validating with data from {source_dir}...")
    metrics = analyzer.validate_from_directory(
        source_dir, test_size=args.test_size, num_workers=args.threads
    )

    print(f"\n{'=' * 60}")
    print("MODEL VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"\nDataset: {metrics['total_samples']} samples")
    print(f"Training set: {metrics['train_samples']} ({metrics['train_pct']:.0%})")
    print(f"Test set: {metrics['test_samples']} ({metrics['test_pct']:.0%})")
    print(f"\nOVERALL ACCURACY: {metrics['accuracy']:.1%}")

    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)
    for class_name, cm in metrics["per_class"].items():
        print(
            f"{class_name:<12} "
            f"{cm['precision']:>10.1%} "
            f"{cm['recall']:>10.1%} "
            f"{cm['f1']:>10.1%} "
            f"{cm['support']:>10}"
        )


def cmd_transcode(args: argparse.Namespace) -> None:
    """Generate training data by transcoding source files."""
    from bitrater.transcode import AudioEncoder

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    encoder = AudioEncoder(source_dir, output_dir)
    encoder.process_files(max_workers=args.workers)


def main() -> None:
    """Entry point for the bitrater CLI."""
    parser = argparse.ArgumentParser(
        prog="bitrater",
        description="Audio quality analysis and bitrate detection",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose output"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="analyze audio files")
    p_analyze.add_argument("target", help="audio file or directory to analyze")
    p_analyze.add_argument("--model", help="path to trained model file")

    # train
    p_train = subparsers.add_parser("train", help="train classifier")
    p_train.add_argument(
        "--source-dir", required=True, help="directory with labeled training data"
    )
    p_train.add_argument("--save-model", help="path to save trained model")
    p_train.add_argument("--threads", type=int, default=None, help="number of workers")

    # validate
    p_validate = subparsers.add_parser("validate", help="validate classifier accuracy")
    p_validate.add_argument(
        "--source-dir", required=True, help="directory with labeled training data"
    )
    p_validate.add_argument(
        "--test-size", type=float, default=0.2, help="fraction for test set (default: 0.2)"
    )
    p_validate.add_argument("--threads", type=int, default=None, help="number of workers")

    # transcode
    p_transcode = subparsers.add_parser(
        "transcode", help="generate training data from source files"
    )
    p_transcode.add_argument(
        "--source-dir", required=True, help="directory with source FLAC/WAV files"
    )
    p_transcode.add_argument(
        "--output-dir", required=True, help="directory for encoded output"
    )
    p_transcode.add_argument("--workers", type=int, default=None, help="number of workers")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    commands = {
        "analyze": cmd_analyze,
        "train": cmd_train,
        "validate": cmd_validate,
        "transcode": cmd_transcode,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
