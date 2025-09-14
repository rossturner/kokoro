#!/usr/bin/env python3
"""
Main Orchestrator Script
Coordinates the complete video dubbing pipeline.
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Optional

from .config import Config, get_default_config
from .constants import SUPPORTED_SAMPLE_RATES, AVAILABLE_VOICES
from .utils import validate_file_exists, setup_logging, format_duration
from .srt_parser import SRTParser
from .sentence_grouper import SentenceGrouper
from .tts_generator import TTSGenerator
from .audio_assembler import AudioAssembler
from .video_processor import VideoProcessor


class DubbingPipeline:
    """Main pipeline for video dubbing process."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.parser = SRTParser()
        self.grouper = SentenceGrouper()
        self.tts_generator = TTSGenerator(config)
        self.assembler = AudioAssembler(config)
        self.video_processor = VideoProcessor(config)

        # Pipeline statistics
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': 0.0,
            'subtitle_entries': 0,
            'sentence_groups': 0,
            'successful_tts_generations': 0,
            'final_audio_duration': 0.0,
            'video_processing_time': 0.0,
            'steps_completed': []
        }

    def process_video(
        self,
        video_path: str,
        srt_path: str,
        output_dir: Optional[str] = None,
        video_name: Optional[str] = None
    ) -> bool:
        """
        Process a complete video through the dubbing pipeline.

        Args:
            video_path: Path to input video file
            srt_path: Path to input SRT subtitle file
            output_dir: Output directory (optional, uses config default)
            video_name: Name for working directory (optional, derived from video)

        Returns:
            True if processing was successful
        """
        self.pipeline_stats['start_time'] = time.time()

        try:
            # Validate inputs
            self._validate_inputs(video_path, srt_path)

            # Setup phase
            self.logger.info("=== Video Dubbing Pipeline ===")
            self.logger.info(f"Input video: {video_path}")
            self.logger.info(f"Input SRT: {srt_path}")
            self.logger.info(f"Configuration: {self.config.voice} voice, {self.config.sample_rate}Hz")

            # Store original video path for output naming
            self.original_video_path = Path(video_path)

            # Create working directory
            if video_name is None:
                video_name = self.original_video_path.stem

            working_dir = self.config.create_working_dir(video_name)
            self.logger.info(f"Working directory: {working_dir}")

            # Phase 1: File setup and validation
            if not self._setup_files(video_path, srt_path, working_dir):
                return False

            # Phase 2: Parse subtitles
            if not self._parse_subtitles(working_dir / "subtitles.srt", working_dir):
                return False

            # Phase 3: Group sentences
            if not self._group_sentences(working_dir):
                return False

            # Phase 4: Generate TTS audio
            if not self._generate_tts_audio(working_dir):
                return False

            # Phase 5: Assemble final audio
            if not self._assemble_audio(working_dir):
                return False

            # Phase 6: Create final video
            if not self._create_final_video(working_dir, output_dir):
                return False

            # Success!
            self._finalize_pipeline(working_dir)
            return True

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            return False
        except Exception as e:
            print(f"Pipeline error: {e}")
            if self.config.verbose_logging:
                import traceback
                traceback.print_exc()
            return False
        finally:
            self.tts_generator.cleanup()

    def _validate_inputs(self, video_path: str, srt_path: str) -> None:
        """Validate input files and configuration."""
        self.logger.debug("Validating inputs")

        # Validate input files exist
        validate_file_exists(Path(video_path), "video file")
        validate_file_exists(Path(srt_path), "SRT file")

        # Validate configuration
        if self.config.voice not in AVAILABLE_VOICES:
            raise ValueError(f"Invalid voice '{self.config.voice}'. Available: {AVAILABLE_VOICES}")

        if self.config.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Invalid sample rate {self.config.sample_rate}Hz. Supported: {SUPPORTED_SAMPLE_RATES}")

        # Check file extensions
        video_ext = Path(video_path).suffix.lower()
        if video_ext not in ['.mp4', '.mkv', '.avi', '.mov', '.wmv']:
            self.logger.warning(f"Unsupported video format: {video_ext}")

        srt_ext = Path(srt_path).suffix.lower()
        if srt_ext != '.srt':
            self.logger.warning(f"Expected .srt file, got: {srt_ext}")

    def _setup_files(self, video_path: str, srt_path: str, working_dir: Path) -> bool:
        """Setup phase: copy and validate input files."""
        print("\n=== Phase 1: File Setup ===")

        try:
            # Validate input files exist
            if not Path(video_path).exists():
                print(f"Error: Video file not found: {video_path}")
                return False

            if not Path(srt_path).exists():
                print(f"Error: SRT file not found: {srt_path}")
                return False

            # Get video information
            video_info = self.video_processor.get_video_info(Path(video_path))
            print(f"Video info: {video_info.duration:.1f}s, {video_info.width}x{video_info.height}, {video_info.fps:.1f}fps")

            if not video_info.has_audio:
                print("Warning: Input video has no audio track")

            # Copy files to working directory (or skip if identical)
            video_copy, srt_copy, copy_status = self.video_processor.copy_files_to_working_dir(
                video_path, srt_path, working_dir
            )

            # Store the copied video path for later use
            self.working_video_path = video_copy

            # Report file setup results
            copied_files = []
            skipped_files = []

            if copy_status['video_copied']:
                copied_files.append('video')
            else:
                skipped_files.append('video')

            if copy_status['srt_copied']:
                copied_files.append('SRT')
            else:
                skipped_files.append('SRT')

            if copied_files:
                print(f"Files copied: {', '.join(copied_files)}")
            if skipped_files:
                print(f"Files skipped (already identical): {', '.join(skipped_files)}")

            print("File setup completed")
            self.pipeline_stats['steps_completed'].append('file_setup')
            return True

        except Exception as e:
            print(f"File setup failed: {e}")
            return False

    def _parse_subtitles(self, srt_path: Path, working_dir: Path) -> bool:
        """Parse SRT subtitles into structured data."""
        print("\n=== Phase 2: Parse Subtitles ===")

        try:
            # Parse SRT file
            self.subtitle_entries = self.parser.parse_file(
                str(srt_path),
                skip_subtitle_instructions=self.config.skip_subtitle_instructions
            )

            print(f"Parsed {len(self.subtitle_entries)} subtitle entries")
            self.pipeline_stats['subtitle_entries'] = len(self.subtitle_entries)

            if len(self.subtitle_entries) == 0:
                print("Error: No subtitle entries found")
                return False

            # Validate entries
            issues = self.parser.validate_entries(self.subtitle_entries)
            if issues:
                print(f"Warning: Found {len(issues)} validation issues:")
                for issue in issues[:5]:  # Show first 5 issues
                    print(f"  - {issue}")

            total_duration = self.parser.get_total_duration(self.subtitle_entries)
            print(f"Total subtitle duration: {total_duration:.1f}s")

            self.pipeline_stats['steps_completed'].append('parse_subtitles')
            return True

        except Exception as e:
            print(f"Subtitle parsing failed: {e}")
            return False

    def _group_sentences(self, working_dir: Path) -> bool:
        """Group subtitle entries into complete sentences."""
        print("\n=== Phase 3: Group Sentences ===")

        try:
            # Group subtitles into sentences
            self.sentence_groups = self.grouper.group_sentences(
                self.subtitle_entries,
                save_debug_info=True,
                debug_path=str(working_dir)
            )

            print(f"Created {len(self.sentence_groups)} sentence groups")
            self.pipeline_stats['sentence_groups'] = len(self.sentence_groups)

            if len(self.sentence_groups) == 0:
                print("Error: No sentence groups created")
                return False

            # Analyze groupings
            analysis = self.grouper.analyze_groupings(self.sentence_groups)
            stats = analysis['statistics']

            print(f"Average duration per sentence: {stats['average_duration']:.2f}s")
            print(f"Average entries per sentence: {stats['average_entries_per_sentence']:.1f}")

            unusual = analysis['unusual_sentences']
            if unusual['long_duration'] > 0:
                print(f"Warning: {unusual['long_duration']} sentences longer than 10 seconds")

            self.pipeline_stats['steps_completed'].append('group_sentences')
            return True

        except Exception as e:
            print(f"Sentence grouping failed: {e}")
            return False

    def _generate_tts_audio(self, working_dir: Path) -> bool:
        """Generate TTS audio for all sentence groups."""
        print("\n=== Phase 4: Generate TTS Audio ===")

        try:
            # Initialize TTS pipeline
            if not self.tts_generator.initialize_pipeline():
                print("Error: Failed to initialize TTS pipeline")
                return False

            # Progress callback
            def progress_callback(current, total, result):
                if current % 10 == 0 or current == total:
                    percent = (current / total) * 100
                    status = "SUCCESS" if result.success else "FAILED"
                    print(f"Progress: {current}/{total} ({percent:.1f}%) - Last: {status}")

            # Generate audio for all sentences
            self.audio_results = self.tts_generator.generate_all_sentences(
                self.sentence_groups,
                working_dir,
                progress_callback=progress_callback
            )

            # Check results
            successful = sum(1 for r in self.audio_results if r.success)
            failed = len(self.audio_results) - successful

            print(f"TTS generation complete: {successful} successful, {failed} failed")
            self.pipeline_stats['successful_tts_generations'] = successful

            if successful == 0:
                print("Error: No TTS audio was generated successfully")
                return False

            if failed > successful * 0.1:  # More than 10% failure rate
                print(f"Warning: High failure rate ({failed/len(self.audio_results)*100:.1f}%)")

            # Print TTS statistics
            self.tts_generator.print_stats()

            self.pipeline_stats['steps_completed'].append('generate_tts')
            return True

        except Exception as e:
            print(f"TTS generation failed: {e}")
            return False

    def _assemble_audio(self, working_dir: Path) -> bool:
        """Assemble individual audio snippets into final audio track."""
        print("\n=== Phase 5: Assemble Audio ===")

        try:
            # Get final audio output path
            final_audio_path = self.config.get_final_audio_path(working_dir)

            # Assemble audio with timing
            success = self.assembler.assemble_audio(
                self.sentence_groups,
                self.audio_results,
                final_audio_path,
                save_timeline_debug=True
            )

            if not success:
                print("Error: Audio assembly failed")
                return False

            # Print assembly statistics
            self.assembler.print_stats()
            stats = self.assembler.get_assembly_stats()
            self.pipeline_stats['final_audio_duration'] = stats.total_duration

            # Analyze timing
            timing_analysis = self.assembler.analyze_timing()
            print(f"\nTiming Analysis:")
            accuracy = timing_analysis['timing_accuracy']
            print(f"  Perfect timing: {accuracy['perfect_timing']} sentences")
            print(f"  Good timing: {accuracy['good_timing']} sentences")
            print(f"  Poor timing: {accuracy['poor_timing']} sentences")

            if timing_analysis['recommendations']:
                print("  Recommendations:")
                for rec in timing_analysis['recommendations']:
                    print(f"    - {rec}")

            self.pipeline_stats['steps_completed'].append('assemble_audio')
            return True

        except Exception as e:
            print(f"Audio assembly failed: {e}")
            return False

    def _create_final_video(self, working_dir: Path, output_dir: Optional[str]) -> bool:
        """Create final dubbed video file."""
        print("\n=== Phase 6: Create Final Video ===")

        try:
            video_path = self.working_video_path  # Use stored copied video file path
            audio_path = self.config.get_final_audio_path(working_dir)
            srt_path = working_dir / "subtitles.srt"  # Copied SRT file

            # Determine output path using original video filename
            output_filename = self.original_video_path.name

            if output_dir:
                output_path = Path(output_dir) / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Use dedicated output directory instead of working directory
                output_path = self.config.output_dir / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Creating final video: {output_path}")

            # Choose video creation method based on configuration
            video_start_time = time.time()

            if self.config.enable_video_compression or self.config.embed_subtitles:
                print("Using enhanced mode (with compression and/or subtitle embedding)...")
                result = self.video_processor.create_dubbed_video_with_compression_and_subtitles(
                    self.working_video_path, audio_path, srt_path, output_path
                )
            else:
                print("Using standard mode (no compression, no subtitle embedding)...")
                result = self.video_processor.create_dubbed_video(
                    video_path, audio_path, output_path
                )

            video_processing_time = time.time() - video_start_time
            self.pipeline_stats['video_processing_time'] = video_processing_time

            if not result.success:
                print(f"Error: Video creation failed: {result.error_message}")
                return False

            print(f"Video created successfully in {video_processing_time:.2f}s")
            print(f"Final video: {output_path}")

            # Validate the result
            validation = self.video_processor.validate_video_audio_sync(
                output_path,
                self.pipeline_stats['final_audio_duration'],
                tolerance=2.0
            )

            if validation['sync_ok']:
                print("Video-audio synchronization validated successfully")
            else:
                print(f"Warning: Sync validation failed - {validation.get('error', 'Duration mismatch')}")

            # Show compression statistics if compression was used
            if self.config.enable_video_compression:
                compression_stats = self.video_processor.get_compression_stats(
                    self.working_video_path, output_path
                )

                if 'error' not in compression_stats:
                    print(f"\nCompression Results:")
                    print(f"  Original size: {compression_stats['original_size_mb']:.1f} MB")
                    print(f"  Compressed size: {compression_stats['compressed_size_mb']:.1f} MB")
                    print(f"  Size reduction: {compression_stats['size_reduction_percent']:.1f}%")
                    print(f"  Quality maintained: {compression_stats['quality_maintained']}")

            # Validate embedded subtitles if enabled
            if self.config.embed_subtitles:
                subtitle_validation = self.video_processor.validate_embedded_subtitles(output_path)
                if subtitle_validation['embedded']:
                    print(f"Subtitles embedded successfully: {subtitle_validation['codec']} format, {subtitle_validation['frame_count']} frames")
                else:
                    print(f"Warning: Subtitle embedding failed - {subtitle_validation.get('error', 'Unknown error')}")

            self.pipeline_stats['steps_completed'].append('create_video')
            return True

        except Exception as e:
            print(f"Video creation failed: {e}")
            return False

    def _finalize_pipeline(self, working_dir: Path) -> None:
        """Finalize pipeline and save final statistics."""
        self.pipeline_stats['end_time'] = time.time()
        self.pipeline_stats['total_duration'] = (
            self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
        )

        # Save pipeline statistics
        stats_path = working_dir / "pipeline_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            # Convert any Path objects to strings for JSON
            json_stats = {}
            for key, value in self.pipeline_stats.items():
                if isinstance(value, Path):
                    json_stats[key] = str(value)
                else:
                    json_stats[key] = value

            json.dump(json_stats, f, indent=2)

        # Print final summary
        print("\n=== Pipeline Complete ===")
        print(f"Total processing time: {self.pipeline_stats['total_duration']:.1f}s")
        print(f"Subtitle entries: {self.pipeline_stats['subtitle_entries']}")
        print(f"Sentence groups: {self.pipeline_stats['sentence_groups']}")
        print(f"Successful TTS generations: {self.pipeline_stats['successful_tts_generations']}")
        print(f"Final audio duration: {self.pipeline_stats['final_audio_duration']:.1f}s")
        print(f"Video processing time: {self.pipeline_stats['video_processing_time']:.1f}s")

        # Cleanup if requested
        if self.config.cleanup_intermediate_files:
            print("Cleaning up intermediate files...")
            self.video_processor.cleanup_working_dir(working_dir, keep_final_files=True)

        print(f"Working directory: {working_dir}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Video Dubbing System using Kokoro TTS with Compression and Subtitle Embedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default: compressed with embedded subtitles)
  python dubbing/main.py --video "video.mp4" --srt "subtitles.srt"

  # High quality archive version
  python dubbing/main.py --video "video.mp4" --srt "subtitles.srt" --compression-preset archive

  # Maximum compression for distribution
  python dubbing/main.py --video "video.mp4" --srt "subtitles.srt" --compression-preset compact

  # Disable compression (original file size)
  python dubbing/main.py --video "video.mp4" --srt "subtitles.srt" --no-compression

  # Custom compression settings
  python dubbing/main.py --video "video.mp4" --srt "subtitles.srt" --video-quality 20 --audio-bitrate 192k
        """
    )

    # Required arguments
    parser.add_argument('--video', required=True,
                       help='Path to input video file')
    parser.add_argument('--srt', required=True,
                       help='Path to input SRT subtitle file')

    # Optional arguments
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for final video (default: ./output/)')
    parser.add_argument('--working-dir', default=None,
                       help='Working directory for processing (default: ./working/[video_name])')
    parser.add_argument('--voice', default='af_heart',
                       choices=['af_heart', 'af_sky', 'af_bella', 'af_nicole', 'am_adam', 'am_michael'],
                       help='Kokoro TTS voice to use (default: af_heart)')

    # Processing options
    parser.add_argument('--mode', choices=['sentence', 'entry'], default='sentence',
                       help='Processing mode: sentence grouping or entry-by-entry (default: sentence)')
    parser.add_argument('--min-gap', type=float, default=0.25,
                       help='Minimum gap between sentences when overflowing (default: 0.25s)')

    # New compression options
    compression_group = parser.add_argument_group('Compression Options')
    compression_group.add_argument('--no-compression', action='store_true',
                                  help='Disable video compression (keep original file size)')
    compression_group.add_argument('--video-quality', type=int, default=None,
                                  help='Video quality (CRF): 0-51, lower=better (default: 23)')
    compression_group.add_argument('--video-preset', default=None,
                                  choices=['ultrafast', 'fast', 'medium', 'slow', 'veryslow'],
                                  help='Encoding speed preset (default: medium)')
    compression_group.add_argument('--audio-bitrate', default=None,
                                  help='Audio compression bitrate (default: 128k)')
    compression_group.add_argument('--compression-preset', default='balanced',
                                  choices=['archive', 'balanced', 'compact'],
                                  help='Quality preset: archive=high quality, balanced=handbrake-like, compact=small files (default: balanced)')

    # New subtitle options
    subtitle_group = parser.add_argument_group('Subtitle Options')
    subtitle_group.add_argument('--no-subtitles', action='store_true',
                               help='Disable subtitle embedding (audio-only dubbing)')
    subtitle_group.add_argument('--subtitle-codec', default=None,
                               choices=['mov_text', 'srt', 'ass', 'webvtt'],
                               help='Subtitle codec for embedding (default: mov_text)')
    subtitle_group.add_argument('--subtitle-language', default=None,
                               help='Subtitle language code (default: eng)')

    # Control options
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up intermediate files after processing')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--config', default=None,
                       help='Path to configuration file (JSON)')

    return parser


def main():
    """Main entry point for the dubbing system."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Setup logging first
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(log_level)
        logger = logging.getLogger(__name__)

        # Load configuration
        if args.config:
            from .config import load_config_from_file
            config = load_config_from_file(args.config)
        else:
            config = get_default_config()

        # Apply command line overrides
        config.voice = args.voice
        config.min_sentence_gap = args.min_gap
        config.sentence_mode = (args.mode == 'sentence')
        config.cleanup_intermediate_files = args.cleanup
        config.verbose_logging = args.verbose

        if args.working_dir:
            config.working_dir = Path(args.working_dir)

        # Apply compression settings
        if args.no_compression:
            config.enable_video_compression = False
            config.embed_subtitles = not args.no_subtitles
        else:
            config.enable_video_compression = True
            config.embed_subtitles = not args.no_subtitles

        # Apply compression preset first, then individual overrides
        if args.compression_preset and not args.no_compression:
            config.apply_compression_preset(args.compression_preset)

        # Individual compression overrides
        if args.video_quality is not None and not args.no_compression:
            config.video_crf = args.video_quality
        if args.video_preset is not None and not args.no_compression:
            config.video_preset = args.video_preset
        if args.audio_bitrate is not None and not args.no_compression:
            config.audio_compression_bitrate = args.audio_bitrate

        # Subtitle settings
        if args.subtitle_codec is not None:
            config.subtitle_codec = args.subtitle_codec
        if args.subtitle_language is not None:
            config.subtitle_language = args.subtitle_language

        # Validate configuration
        issues = config.validate()
        if issues:
            print("Configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            return 1

        if config.verbose_logging:
            print(f"Configuration:\n{config}")

        # Create and run pipeline
        pipeline = DubbingPipeline(config)
        success = pipeline.process_video(
            args.video,
            args.srt,
            args.output_dir,
            Path(args.video).stem
        )

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())