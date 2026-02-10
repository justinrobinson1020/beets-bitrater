"""Script for generating training data by transcoding audio files."""

import argparse
import concurrent.futures
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

# # Add parent directory to Python path to allow absolute imports
# sys.path.insert(0, str(Path(__file__).parent.parent))
from bitrater.constants import CBR_BITRATES, VBR_PRESETS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"encoding_log_{datetime.now():%Y%m%d_%H%M%S}.txt"),
    ],
)
logger = logging.getLogger(__name__)


class AudioEncoder:
    """Handles encoding of audio files for training data generation."""

    def __init__(self, source_dir: Path, output_dir: Path):
        """
        Initialize encoder with source and output directories.

        Args:
            source_dir: Directory containing source audio files
            output_dir: Directory for encoded output files
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verify required executables
        self.executables = self._find_executables()

        # Track processed files to avoid duplicates
        self.processed_files: set[str] = set()

    def _find_executables(self) -> dict[str, str]:
        """Find required executable paths."""
        executables = {
            "lame": shutil.which("lame"),
            "flac": shutil.which("flac"),
            "ffmpeg": shutil.which("ffmpeg"),
        }

        missing = [name for name, path in executables.items() if not path]
        if missing:
            raise RuntimeError(
                f"Required executables not found: {', '.join(missing)}. "
                "Please install LAME, FLAC, and FFmpeg."
            )

        return executables

    def process_files(self, max_workers: int | None = None, include_uptranscoding: bool = True) -> None:
        """
        Process all audio files in source directory with optional uptranscoding.

        Args:
            max_workers: Maximum number of worker threads (None = CPU count - 1)
            include_uptranscoding: Whether to create uptranscoded files after MP3 encoding
        """
        # Check if migration is needed
        self.check_migration_needed()

        # Stage 1: Create MP3 files
        self._create_mp3_files(max_workers)

        # Stage 2: Create uptranscoded FLAC files from MP3s
        if include_uptranscoding:
            logger.info("\n" + "="*50)
            logger.info("STAGE 2: Creating uptranscoded FLAC files")
            logger.info("="*50)
            self.create_uptranscoded_files(max_workers)

    def _create_mp3_files(self, max_workers: int | None = None) -> None:
        """Create MP3 files from source FLAC/WAV files (Stage 1).

        Uses a single global thread pool across all files so that --workers N
        can keep N threads busy even when N exceeds the number of encoding
        formats per file.
        """
        source_files = self._collect_source_files()
        if not source_files:
            raise ValueError(f"No source files found in {self.source_dir}")

        total_formats = len(CBR_BITRATES) + len(VBR_PRESETS)
        progress = ProgressTracker(len(source_files), total_formats, phase_name="MP3 Encoding")

        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)

        logger.info("STAGE 1: MP3 encoding process:")
        logger.info(f"├── Found {len(source_files)} source files")
        logger.info(f"├── Will create {total_formats} formats per file")
        logger.info(
            f"├── Total encodes to perform: {len(source_files) * total_formats}"
        )
        logger.info(f"└── Using {max_workers} worker threads")

        # Cap concurrent in-flight WAVs to limit disk usage while keeping
        # the thread pool fed.  We want enough files queued so workers
        # never starve while the main thread decodes the next WAV.
        # 2 * ceil(workers/formats) keeps a full batch queued while
        # the current batch drains.
        max_inflight = max(3, -(-max_workers // total_formats) * 2)

        # Each entry: (wav_path, futures_list, filename, task_count)
        in_flight: list[
            tuple[Path | None, list[concurrent.futures.Future], str, int]
        ] = []

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                for source_file in source_files:
                    # Drain completed files to free WAVs and keep slots open
                    self._drain_ready_files(in_flight, progress)
                    # If still at capacity, block on the oldest
                    while len(in_flight) >= max_inflight:
                        self._drain_completed_file(in_flight, progress)

                    sanitized_name = self._sanitize_filename(source_file.stem)

                    # Ensure lossless symlink exists for every source file
                    self._ensure_lossless_symlink(source_file, sanitized_name)

                    try:
                        prepared = self._prepare_file(
                            source_file, sanitized_name
                        )
                    except Exception as e:
                        logger.error(f"Error preparing {source_file.name}: {e}")
                        continue

                    if prepared is None:
                        continue

                    temp_wav = prepared
                    tasks = self._create_encoding_tasks(temp_wav, sanitized_name)

                    if not tasks:
                        logger.debug(f"  {source_file.name}: all formats exist, skipping")
                        self._cleanup_temp_file(temp_wav)
                        continue

                    futures = [
                        executor.submit(self._run_encode, task) for task in tasks
                    ]
                    in_flight.append(
                        (temp_wav, futures, source_file.name, len(tasks))
                    )

                # Drain remaining in-flight files
                while in_flight:
                    self._drain_completed_file(in_flight, progress)

        except KeyboardInterrupt:
            logger.warning("\nMP3 encoding interrupted by user")
            for wav_path, futures, _, _ in in_flight:
                for f in futures:
                    f.cancel()
                self._cleanup_temp_file(wav_path)
            raise

        finally:
            progress.finish()

    def _collect_source_files(self) -> list[Path]:
        """Collect all valid source audio files."""
        logger.info("Scanning for source files...")
        source_files = []
        for ext in [".flac", ".wav"]:
            source_files.extend(self.source_dir.glob(f"**/*{ext}"))
        logger.info(f"Found {len(source_files)} audio files")
        return source_files

    def _prepare_file(self, source_file: Path, sanitized_name: str) -> Path | None:
        """Decode source file to WAV and return the wav path for encoding.

        Returns:
            temp_wav_path, or None if all outputs already exist.
        """
        if self._check_outputs_exist(sanitized_name):
            logger.debug(f"Skipping {source_file.name} - already encoded")
            return None

        # Create temporary WAV in output dir (avoids writes to slow source disk)
        temp_dir = self.output_dir / ".tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_wav = temp_dir / f"temp_{source_file.stem}.wav"

        logger.debug(f"Decoding: {source_file.name}")

        if source_file.suffix.lower() == ".flac":
            self._decode_flac(source_file, temp_wav)
        else:
            temp_wav = source_file
            logger.debug(f"  Using existing WAV: {source_file.name}")

        return temp_wav

    def _ensure_lossless_symlink(self, source_file: Path, sanitized_name: str) -> None:
        """Create a symlink in lossless/ pointing to the original source file."""
        lossless_dir = self.output_dir / "lossless"
        lossless_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = lossless_dir / f"{sanitized_name}{source_file.suffix}"
        if not symlink_path.exists():
            try:
                symlink_path.symlink_to(source_file.resolve())
            except OSError as e:
                logger.error(f"Failed to create lossless symlink for {source_file.name}: {e}")

    def _drain_completed_file(
        self,
        in_flight: list[
            tuple["Path | None", list[concurrent.futures.Future], str, int]
        ],
        progress: "ProgressTracker",
    ) -> None:
        """Wait for the oldest in-flight file to finish, update progress, clean up."""
        wav_path, futures, filename, task_count = in_flight.pop(0)
        try:
            successful = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    if future.result():
                        successful += 1
                except Exception as e:
                    logger.error(f"Encoding task failed for {filename}: {e}")
                progress.complete_task()
            progress.complete_file(filename, task_count, successful)
        finally:
            self._cleanup_temp_file(wav_path)

    def _drain_ready_files(
        self,
        in_flight: list[
            tuple["Path | None", list[concurrent.futures.Future], str, int]
        ],
        progress: "ProgressTracker",
    ) -> None:
        """Non-blocking drain: collect any in-flight files whose futures all finished."""
        i = 0
        while i < len(in_flight):
            wav_path, futures, filename, task_count = in_flight[i]
            if all(f.done() for f in futures):
                in_flight.pop(i)
                successful = sum(
                    1 for f in futures
                    if not f.exception() and f.result()
                )
                for f in futures:
                    if f.exception():
                        logger.error(f"Encoding task failed for {filename}: {f.exception()}")
                    progress.complete_task()
                progress.complete_file(filename, task_count, successful)
                self._cleanup_temp_file(wav_path)
            else:
                i += 1

    def _decode_flac(self, source_file: Path, output_path: Path) -> None:
        """Decode FLAC to WAV format."""
        start_time = time.time()
        try:
            cmd = [
                self.executables["flac"],
                "--decode",
                "--totally-silent",  # Reduce noise in logs
                str(source_file),
                "--output-name",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            wav_time = time.time() - start_time
            logger.debug(f"WAV decode: {source_file.name} in {wav_time:.1f}s")

        except subprocess.CalledProcessError as e:
            logger.error(f"└── FLAC decoding failed: {e.stderr}")
            raise

    def _create_encoding_tasks(
        self, wav_path: Path, source_name: str
    ) -> list[tuple[list[str], Path, str]]:
        """Create list of encoding tasks for parallel processing."""
        tasks = []

        # Add CBR tasks (now under lossy/ subdirectory)
        for bitrate in CBR_BITRATES:
            output_path = self.output_dir / "lossy" / str(bitrate) / f"{source_name}.mp3"
            old_path = self.output_dir / str(bitrate) / f"{source_name}.mp3"

            # Skip if file exists in either new or old location
            if output_path.exists() or old_path.exists():
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                self.executables["lame"],
                "--silent",
                "--cbr",
                "-b",
                str(bitrate),
                "-q",
                "0" if bitrate >= 256 else "2",  # High quality for high bitrates
                "--replaygain-accurate",  # Enable accurate ReplayGain analysis
                str(wav_path),
                str(output_path),
            ]
            tasks.append((cmd, output_path, f"CBR-{bitrate}"))

        # Add VBR tasks (now under lossy/ subdirectory)
        for preset in VBR_PRESETS:
            output_path = self.output_dir / "lossy" / f"v{preset}" / f"{source_name}.mp3"
            old_path = self.output_dir / f"v{preset}" / f"{source_name}.mp3"

            # Skip if file exists in either new or old location
            if output_path.exists() or old_path.exists():
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                self.executables["lame"],
                "--silent",
                "-V",
                str(preset),
                "--vbr-new",
                "--replaygain-accurate",
            ]

            # Add high quality flags for V0/V1
            if preset in [0, 1]:
                cmd.extend(["-q", "0", "-h"])

            cmd.extend([str(wav_path), str(output_path)])
            tasks.append((cmd, output_path, f"VBR-{preset}"))

        return tasks

    def _run_encode(self, task: tuple[list[str], Path, str]) -> bool:
        """Execute a single encoding task."""
        cmd, output_path, task_id = task
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Error encoding {task_id}: {e.stderr}")
            if output_path.exists():
                output_path.unlink()
            return False

        except Exception as e:
            logger.error(f"Exception encoding {task_id}: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            return False

    def _check_outputs_exist(self, source_name: str) -> bool:
        """Check if all output formats exist for source in either old or new location."""
        all_exist = True

        for bitrate in CBR_BITRATES:
            # Check new location first
            new_path = self.output_dir / "lossy" / str(bitrate) / f"{source_name}.mp3"
            # Check old location as fallback
            old_path = self.output_dir / str(bitrate) / f"{source_name}.mp3"

            if not (new_path.exists() or old_path.exists()):
                all_exist = False
                break

        if all_exist:
            for preset in VBR_PRESETS:
                # Check new location first
                new_path = self.output_dir / "lossy" / f"v{preset}" / f"{source_name}.mp3"
                # Check old location as fallback
                old_path = self.output_dir / f"v{preset}" / f"{source_name}.mp3"

                if not (new_path.exists() or old_path.exists()):
                    all_exist = False
                    break

        return all_exist

    def _cleanup_temp_file(self, temp_path: Path | None) -> None:
        """Clean up temporary WAV file."""
        if temp_path and temp_path.exists() and temp_path.name.startswith("temp_"):
            try:
                temp_path.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_path.name}")
            except Exception as e:
                logger.error(f"Error cleaning up {temp_path.name}: {str(e)}")

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent command-line issues."""
        # Normalize unicode characters
        filename = unicodedata.normalize("NFKD", filename)
        # Replace problematic characters
        filename = re.sub(r"[\[\]\(\)\{\}]", "_", filename)
        # Remove non-word characters (except dashes and dots)
        filename = re.sub(r"[^\w\-\.]", "_", filename)
        # Collapse multiple underscores
        filename = re.sub(r"_+", "_", filename)
        return filename.strip("_")

    def check_migration_needed(self) -> bool:
        """Check if migration from old to new directory structure is needed."""
        old_dirs_with_files = []

        # Check for files in old directory structure
        for bitrate in CBR_BITRATES:
            old_dir = self.output_dir / str(bitrate)
            if old_dir.exists() and list(old_dir.glob("*.mp3")):
                old_dirs_with_files.append(str(bitrate))

        for preset in VBR_PRESETS:
            old_dir = self.output_dir / f"v{preset}"
            if old_dir.exists() and list(old_dir.glob("*.mp3")):
                old_dirs_with_files.append(f"v{preset}")

        if old_dirs_with_files:
            logger.info("\nMIGRATION NOTICE:")
            logger.info(f"├── Found files in old directory structure: {old_dirs_with_files}")
            logger.info("├── Consider migrating to new structure for better organization")
            logger.info("├── Preview: python transcode.py --migrate --dry-run")
            logger.info("└── Migrate: python transcode.py --migrate")
            return True

        return False

    def create_uptranscoded_files(self, max_workers: int | None = None) -> None:
        """
        Create uptranscoded FLAC files from MP3 files.

        This creates MP3→FLAC conversions for training the uptranscode detector.

        Args:
            max_workers: Maximum number of worker threads (None = CPU count - 1)
        """
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)

        # Find all MP3 files in lossy/ subdirectory
        mp3_files = self._collect_mp3_files()
        if not mp3_files:
            logger.warning("No MP3 files found for uptranscoding. Run regular encoding first.")
            return

        total_files = len(mp3_files)
        logger.info("Starting uptranscode generation:")
        logger.info(f"├── Found {total_files} MP3 files")
        logger.info(f"└── Using {max_workers} worker threads")

        progress = ProgressTracker(total_files, 1, phase_name="Uptranscode")

        try:
            # Process in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._uptranscode_file, mp3_file, progress)
                          for mp3_file in mp3_files]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Uptranscode task failed: {str(e)}")

        except KeyboardInterrupt:
            logger.warning("\nUptranscode process interrupted by user")
            raise

        finally:
            progress.finish()

    def _collect_mp3_files(self) -> list[Path]:
        """Collect all MP3 files from both old and new directory structures."""
        mp3_files = []

        # Check new lossy/ subdirectory first
        lossy_dir = self.output_dir / "lossy"
        old_format_dirs = []

        # Collect from new directory structure
        if lossy_dir.exists():
            for bitrate in CBR_BITRATES:
                bitrate_dir = lossy_dir / str(bitrate)
                if bitrate_dir.exists():
                    mp3_files.extend(bitrate_dir.glob("*.mp3"))

            for preset in VBR_PRESETS:
                preset_dir = lossy_dir / f"v{preset}"
                if preset_dir.exists():
                    mp3_files.extend(preset_dir.glob("*.mp3"))

        # Also check old directory structure for files that haven't been migrated yet
        for bitrate in CBR_BITRATES:
            old_dir = self.output_dir / str(bitrate)
            if old_dir.exists():
                old_files = list(old_dir.glob("*.mp3"))
                if old_files:
                    logger.warning(f"Found {len(old_files)} files in old location: {old_dir}")
                    logger.warning("Consider running migration: python transcode.py --migrate")
                    mp3_files.extend(old_files)
                    old_format_dirs.append(str(old_dir))

        for preset in VBR_PRESETS:
            old_dir = self.output_dir / f"v{preset}"
            if old_dir.exists():
                old_files = list(old_dir.glob("*.mp3"))
                if old_files:
                    logger.warning(f"Found {len(old_files)} files in old location: {old_dir}")
                    if str(old_dir) not in old_format_dirs:  # Avoid duplicate warning
                        logger.warning("Consider running migration: python transcode.py --migrate")
                    mp3_files.extend(old_files)

        if old_format_dirs:
            logger.info("Collected MP3 files from both old and new directory structures")

        return mp3_files

    def _uptranscode_file(self, mp3_file: Path, progress: "ProgressTracker") -> None:
        """Uptranscode a single MP3 file to FLAC."""
        # Determine source format from path
        parent_name = mp3_file.parent.name
        if parent_name.startswith('v'):
            source_format = f"from_{parent_name}"
        else:
            source_format = f"from_{parent_name}"

        # Create output path
        uptranscode_dir = self.output_dir / "uptranscoded" / source_format
        uptranscode_dir.mkdir(parents=True, exist_ok=True)

        output_file = uptranscode_dir / f"{mp3_file.stem}.flac"

        # Skip if already exists
        if output_file.exists():
            logger.debug(f"Skipping existing uptranscode: {output_file.name}")
            return

        # Convert MP3 to FLAC using FFmpeg
        try:
            cmd = [
                self.executables["ffmpeg"],
                "-loglevel", "error",
                "-i", str(mp3_file),
                "-c:a", "flac",
                "-compression_level", "8",  # High compression
                "-y",  # Overwrite output files
                str(output_file)
            ]

            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.debug(f"Uptranscoded: {mp3_file.name} → {output_file.name}")
            progress.update_file_progress(1, 1, mp3_file.name)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to uptranscode {mp3_file.name}: {e.stderr}")
            # Clean up partial file
            if output_file.exists():
                output_file.unlink()
            raise

        except Exception as e:
            logger.error(f"Error uptranscoding {mp3_file.name}: {str(e)}")
            if output_file.exists():
                output_file.unlink()
            raise


class ProgressTracker:
    """Track progress and timing of the encoding process."""

    def __init__(self, total_files: int, total_formats: int, phase_name: str = "Encoding"):
        self.total_files = total_files
        self.total_formats = total_formats
        self.total_tasks = total_files * total_formats
        self.completed_files = 0
        self.completed_tasks = 0
        self.start_time = time.time()
        self.phase_name = phase_name

    def complete_task(self) -> None:
        """Record one encoding task completed."""
        self.completed_tasks += 1

    def complete_file(
        self, filename: str, task_count: int, successful: int
    ) -> None:
        """Record a file fully completed and log progress."""
        self.completed_files += 1
        elapsed = time.time() - self.start_time
        overall_pct = (
            (self.completed_tasks / self.total_tasks * 100)
            if self.total_tasks
            else 0
        )
        files_per_sec = self.completed_files / elapsed if elapsed > 0 else 0
        remaining = self.total_files - self.completed_files
        eta = remaining / files_per_sec if files_per_sec > 0 else 0

        logger.info(
            f"Completed {filename}: {successful}/{task_count} formats | "
            f"Overall: {self.completed_files}/{self.total_files} files ({overall_pct:.0f}%) | "
            f"ETA: {timedelta(seconds=int(eta))}"
        )

    def update_file_progress(
        self, completed_tasks: int, total_file_tasks: int, filename: str
    ) -> None:
        """Update progress for current file (used by uptranscode phase)."""
        self.completed_tasks += completed_tasks
        elapsed = time.time() - self.start_time
        overall_pct = (
            (self.completed_tasks / self.total_tasks * 100)
            if self.total_tasks
            else 0
        )
        tasks_per_sec = self.completed_tasks / elapsed if elapsed > 0 else 0
        remaining = self.total_tasks - self.completed_tasks
        eta = remaining / tasks_per_sec if tasks_per_sec > 0 else 0

        logger.info(
            f"Uptranscoded {filename} | "
            f"{self.completed_tasks}/{self.total_tasks} ({overall_pct:.0f}%) | "
            f"ETA: {timedelta(seconds=int(eta))}"
        )

    def finish(self) -> None:
        """Log final statistics."""
        total_time = time.time() - self.start_time
        processed = self.completed_files if self.completed_files > 0 else self.total_files
        avg_time_per_file = total_time / processed if processed > 0 else 0

        logger.info(f"\n{self.phase_name} Summary:")
        logger.info("=" * 50)
        logger.info(f"Total Files Processed: {self.total_files}")
        logger.info(f"Total Formats Per File: {self.total_formats}")
        logger.info(f"Total Encodes Completed: {self.completed_tasks}")
        logger.info(f"Total Time: {timedelta(seconds=int(total_time))}")
        logger.info(
            f"Average Time Per File: {timedelta(seconds=int(avg_time_per_file))}"
        )
        logger.info("=" * 50)


def main() -> None:
    """Main entry point for transcoding script."""
    parser = argparse.ArgumentParser(
        description="Generate training data for bitrater plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate existing files to new structure
  python transcode.py --migrate

  # Preview migration (dry run)
  python transcode.py --migrate --dry-run

  # Create MP3 files only (Stage 1)
  python transcode.py --no-uptranscode

  # Create MP3 files and uptranscoded FLAC files (both stages)
  python transcode.py

  # Create only uptranscoded files (assuming MP3s already exist)
  python transcode.py --uptranscode-only
        """
    )

    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate existing files from old directory structure to new structure"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --migrate, show what would be moved without actually moving files"
    )

    parser.add_argument(
        "--no-uptranscode",
        action="store_true",
        help="Skip uptranscode generation (MP3→FLAC conversion)"
    )

    parser.add_argument(
        "--uptranscode-only",
        action="store_true",
        help="Only create uptranscoded files (skip MP3 generation)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: CPU cores - 1)"
    )

    args = parser.parse_args()

    # Handle migration mode
    if args.migrate:
        try:
            output_dir = Path("encoded")
            if not output_dir.exists():
                raise ValueError(f"Output directory '{output_dir}' does not exist")

            migrate_existing_files(output_dir, dry_run=args.dry_run)
            return  # Exit after migration

        except KeyboardInterrupt:
            logger.warning("\nMigration interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            sys.exit(1)

    # Regular transcoding mode
    try:
        source_dir = Path("lossless")
        output_dir = Path("encoded")

        if not source_dir.exists() and not args.uptranscode_only:
            raise ValueError(f"Source directory '{source_dir}' does not exist")

        # Clean up any leftover temporary files (check both old and new locations)
        for temp_parent in [source_dir, output_dir / ".tmp"]:
            if temp_parent.exists():
                temp_files = list(temp_parent.glob("temp_*.wav"))
                if temp_files:
                    logger.info(f"Cleaning up {len(temp_files)} temporary files in {temp_parent}...")
                    for temp_file in temp_files:
                        try:
                            temp_file.unlink()
                        except Exception as e:
                            logger.error(f"Error deleting {temp_file}: {e}")

        # Initialize encoder
        encoder = AudioEncoder(source_dir, output_dir)

        # Determine number of workers
        max_workers = args.workers if args.workers else max(1, os.cpu_count() - 1)

        # Execute based on options
        if args.uptranscode_only:
            logger.info("Running uptranscode-only mode")
            encoder.create_uptranscoded_files(max_workers)
        else:
            # Default mode: create MP3s and optionally uptranscoded files
            include_uptranscoding = not args.no_uptranscode
            encoder.process_files(max_workers, include_uptranscoding)

    except KeyboardInterrupt:
        logger.warning("\nEncoding process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Encoding process failed: {e}")
        sys.exit(1)


def migrate_existing_files(output_dir: Path, dry_run: bool = False) -> None:
    """
    Migrate existing training data files from old structure to new structure.

    Old structure: encoded/{128,192,256,320,v0,v2,v4}/
    New structure: encoded/lossy/{128,192,256,320,v0,v2,v4}/

    Args:
        output_dir: Base output directory (usually 'encoded')
        dry_run: If True, only show what would be moved without actually moving
    """
    logger.info("=" * 60)
    logger.info("MIGRATION: Moving existing files to new directory structure")
    logger.info("=" * 60)

    # Get all bitrate directories and VBR preset directories
    old_dirs = []
    for bitrate in CBR_BITRATES:
        old_dir = output_dir / str(bitrate)
        if old_dir.exists():
            old_dirs.append((old_dir, str(bitrate)))

    for preset in VBR_PRESETS:
        old_dir = output_dir / f"v{preset}"
        if old_dir.exists():
            old_dirs.append((old_dir, f"v{preset}"))

    if not old_dirs:
        logger.info("No old-format directories found. Migration not needed.")
        return

    total_files = 0
    for old_dir, _ in old_dirs:
        mp3_files = list(old_dir.glob("*.mp3"))
        total_files += len(mp3_files)

    if total_files == 0:
        logger.info("No MP3 files found in old directories.")
        return

    logger.info(f"Found {len(old_dirs)} old directories with {total_files} total files")

    if dry_run:
        logger.info("DRY RUN MODE - No files will be moved")

    # Create new lossy directory
    lossy_dir = output_dir / "lossy"
    if not dry_run:
        lossy_dir.mkdir(exist_ok=True)

    moved_files = 0
    skipped_files = 0

    for old_dir, format_name in old_dirs:
        logger.info(f"\nProcessing {format_name} directory...")

        # Create new directory structure
        new_dir = lossy_dir / format_name
        if not dry_run:
            new_dir.mkdir(exist_ok=True)

        # Move all MP3 files
        mp3_files = list(old_dir.glob("*.mp3"))
        logger.info(f"├── Found {len(mp3_files)} MP3 files in {old_dir}")

        for mp3_file in mp3_files:
            new_path = new_dir / mp3_file.name

            if new_path.exists():
                logger.warning(f"├── SKIP: {mp3_file.name} (already exists in new location)")
                skipped_files += 1
                continue

            if dry_run:
                logger.info(f"├── WOULD MOVE: {mp3_file.name}")
            else:
                try:
                    # Move the file
                    mp3_file.rename(new_path)
                    logger.debug(f"├── MOVED: {mp3_file.name}")
                    moved_files += 1
                except Exception as e:
                    logger.error(f"├── ERROR moving {mp3_file.name}: {e}")

        # Remove old directory if empty
        if not dry_run and old_dir.exists():
            try:
                # Check if directory is empty
                remaining_files = list(old_dir.iterdir())
                if not remaining_files:
                    old_dir.rmdir()
                    logger.info(f"└── Removed empty directory: {old_dir}")
                else:
                    logger.warning(f"└── Left directory {old_dir} (contains {len(remaining_files)} files)")
            except Exception as e:
                logger.error(f"└── Error removing directory {old_dir}: {e}")

    logger.info("\nMigration Summary:")
    logger.info(f"├── Files moved: {moved_files}")
    logger.info(f"├── Files skipped: {skipped_files}")
    logger.info(f"└── Total processed: {total_files}")

    if dry_run:
        logger.info("\nTo perform actual migration, run: python transcode.py --migrate")
    else:
        logger.info("\nMigration completed successfully!")


def main_migrate() -> None:
    """Entry point for migration-only operation."""
    parser = argparse.ArgumentParser(
        description="Migrate existing training data to new directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files"
    )

    args = parser.parse_args()

    try:
        output_dir = Path("encoded")
        if not output_dir.exists():
            raise ValueError(f"Output directory '{output_dir}' does not exist")

        migrate_existing_files(output_dir, dry_run=args.dry_run)

    except KeyboardInterrupt:
        logger.warning("\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
