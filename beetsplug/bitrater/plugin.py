"""Plugin interface for beets-bitrater."""

import logging
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from beets import util
from beets.dbcore import types
from beets.library import Item, Library
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, UserError, decargs

from .analyzer import AudioQualityAnalyzer
from .types import AnalysisResult

logger = logging.getLogger("beets.bitrater")


class BitraterPlugin(BeetsPlugin):
    """Plugin for analyzing audio quality and detecting transcodes."""

    def __init__(self) -> None:
        super().__init__()
        self.analyzer = AudioQualityAnalyzer()
        self.config.add(
            {
                "auto": False,  # Run automatically on import
                "min_confidence": 0.7,  # Minimum confidence threshold
                "warn_transcodes": True,  # Warn about detected transcodes
                "threads": None,  # Number of analysis threads (None = auto)
                "model_path": None,  # Path to trained model
                "training_dir": None,  # Path to training data directory
            }
        )

        # Add new fields to the database
        self.item_types = {
            "original_bitrate": types.INTEGER,
            "original_format": types.STRING,
            "bitrate_confidence": types.FLOAT,
            "is_transcoded": types.BOOLEAN,
            "transcoded_from": types.STRING,
            "analysis_version": types.STRING,
            "analysis_date": types.STRING,
            "format_warnings": types.STRING,
        }

        # Load trained model if configured
        model_path = self.config["model_path"].get()
        if model_path:
            try:
                self.analyzer.load_model(Path(model_path))
                logger.info(f"Loaded classifier model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load classifier model: {e}")

        # Register import listener
        self.register_listener("import_task_files", self.import_task)

    def commands(self) -> list[Subcommand]:
        """Create plugin commands."""
        analyze_cmd = Subcommand(
            "bitrater", help="Analyze audio files to detect original bitrate"
        )
        analyze_cmd.parser.add_option(
            "-v", "--verbose", action="store_true", help="show detailed analysis results"
        )
        analyze_cmd.parser.add_option(
            "-t", "--train", action="store_true", help="train classifier with known-good files"
        )
        analyze_cmd.parser.add_option(
            "--threads", type="int", help="number of analysis threads"
        )
        analyze_cmd.parser.add_option(
            "--save-model", help="save trained model to specified path"
        )
        analyze_cmd.parser.add_option(
            "--training-dir", help="path to training data directory"
        )
        analyze_cmd.func = self.analyze_command
        return [analyze_cmd]

    def analyze_command(self, lib: Library, opts: Any, args: list[str]) -> None:
        """Handle the analyze command."""
        try:
            if opts.train:
                self._train_classifier(opts)
                return

            # Check if model is trained
            if not self.analyzer.is_trained:
                logger.warning(
                    "Classifier not trained. Run with --train first or specify --model-path"
                )

            # Get items to analyze
            items = lib.items(decargs(args)) if args else lib.items()
            items = list(items)  # Materialize query

            if not items:
                logger.info("No items to analyze")
                return

            # Configure threading
            thread_count = opts.threads or self.config["threads"].get() or util.cpu_count()

            # Analyze files
            results = self._analyze_items(items, thread_count)
            self._process_results(items, results, opts.verbose)

        except Exception as e:
            raise UserError(f"Analysis failed: {e}") from e

    def _analyze_items(
        self, items: Sequence[Item], thread_count: int
    ) -> list[AnalysisResult | None]:
        """Analyze multiple items in parallel."""
        logger.info(f"Analyzing {len(items)} files using {thread_count} threads")

        results: list[AnalysisResult | None] = [None] * len(items)
        lock = threading.Lock()
        progress = {"done": 0}
        total = len(items)

        def analyze_item(index_item: tuple) -> None:
            index, item = index_item
            try:
                result = self.analyzer.analyze_file(str(item.path))
                results[index] = result
            except Exception as e:
                logger.error(f"Error analyzing {item.path}: {e}")
                results[index] = None

            with lock:
                progress["done"] += 1
                done = progress["done"]
                if done % 10 == 0 or done == total:
                    logger.info(f"Analyzed {done}/{total} files")

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            executor.map(analyze_item, enumerate(items))

        return results

    def _process_results(
        self,
        items: Sequence[Item],
        results: Sequence[AnalysisResult | None],
        verbose: bool,
    ) -> None:
        """Process and store analysis results."""
        total = 0
        transcodes = 0
        low_confidence = 0

        min_confidence = self.config["min_confidence"].get()
        warn_transcodes = self.config["warn_transcodes"].get()

        for item, result in zip(items, results, strict=True):
            if not result:
                continue

            total += 1

            # Update item with analysis results
            item.original_bitrate = result.original_bitrate
            item.original_format = result.original_format
            item.bitrate_confidence = result.confidence
            item.is_transcoded = result.is_transcode
            item.transcoded_from = result.transcoded_from or ""
            item.analysis_version = result.analysis_version
            item.analysis_date = result.analysis_date.isoformat()
            item.format_warnings = "; ".join(result.warnings)

            # Update statistics
            if result.is_transcode:
                transcodes += 1
                if warn_transcodes:
                    logger.warning(
                        f"Transcode detected: {item.path} "
                        f"(appears to be {result.transcoded_from})"
                    )
            if result.confidence < min_confidence:
                low_confidence += 1

            if verbose:
                self._print_analysis(item, result)

            item.store()

        # Print summary
        self._print_summary(total, transcodes, low_confidence)

    def _train_classifier(self, opts: Any) -> None:
        """Train classifier using known-good files from training directory."""
        # Get training directory from options or config
        training_dir_str = opts.training_dir or self.config["training_dir"].get()
        if not training_dir_str:
            raise UserError(
                "Training directory not specified. "
                "Use --training-dir or set training_dir in config"
            )

        training_dir = Path(training_dir_str)
        if not training_dir.exists():
            raise UserError(f"Training directory not found: {training_dir}")

        logger.info(f"Training classifier from {training_dir}...")

        try:
            # Train using directory structure
            stats = self.analyzer.train_from_directory(training_dir)

            logger.info(
                f"Training complete: {stats['successful']}/{stats['total_files']} files processed"
            )

            # Save model if requested
            if opts.save_model:
                save_path = Path(opts.save_model)
                self.analyzer.save_model(save_path)
                logger.info(f"Saved trained model to {save_path}")

        except Exception as e:
            raise UserError(f"Training failed: {e}") from e

    def _print_analysis(self, item: Item, result: AnalysisResult) -> None:
        """Print detailed analysis results for an item."""
        print(f"\n{item.title}")
        print("-" * 50)
        print(f"Path: {item.path}")
        print(f"File format: {result.file_format}")
        print(f"Stated bitrate: {result.stated_bitrate or 'N/A'} kbps")
        print(f"Detected original: {result.original_format} ({result.original_bitrate} kbps)")
        print(f"Confidence: {result.confidence:.1%}")

        if result.is_transcode:
            print(f"⚠️  TRANSCODE DETECTED - appears to be from {result.transcoded_from}")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

    def _print_summary(self, total: int, transcodes: int, low_confidence: int) -> None:
        """Print analysis summary."""
        print("\n" + "=" * 50)
        print("Analysis Summary")
        print("=" * 50)
        print(f"Total files analyzed: {total}")

        if transcodes > 0:
            print(f"⚠️  Potential transcodes detected: {transcodes}")
        if low_confidence > 0:
            print(f"⚠️  Low confidence results: {low_confidence}")

        if transcodes == 0 and low_confidence == 0:
            print("✓ All files appear to be original quality")

    def import_task(self, session: Any, task: Any) -> None:
        """Automatically analyze files during import if enabled.

        Args:
            session: Beets import session (beets.importer.ImportSession)
            task: Beets import task (beets.importer.ImportTask)
        """
        if not self.config["auto"].get():
            return

        if not self.analyzer.is_trained:
            return

        for item in task.items:
            try:
                result = self.analyzer.analyze_file(str(item.path))
                if result:
                    # Update item fields
                    item.original_bitrate = result.original_bitrate
                    item.original_format = result.original_format
                    item.bitrate_confidence = result.confidence
                    item.is_transcoded = result.is_transcode
                    item.transcoded_from = result.transcoded_from or ""
                    item.analysis_version = result.analysis_version
                    item.analysis_date = result.analysis_date.isoformat()
                    item.format_warnings = "; ".join(result.warnings)

                    if result.is_transcode and self.config["warn_transcodes"].get():
                        logger.warning(
                            f"Import: Transcode detected - {item.path} "
                            f"(appears to be {result.transcoded_from})"
                        )
            except Exception as e:
                logger.error(f"Auto-analysis failed for {item.path}: {e}")
