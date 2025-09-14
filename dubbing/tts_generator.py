"""
TTS Generator Module
Generates audio snippets using Kokoro TTS for sentence groups.
"""

import logging
import time
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Callable
from dataclasses import dataclass

from kokoro import KPipeline
from .sentence_grouper import SentenceGroup
from .config import Config
from .constants import (
    AUDIO_SUBTYPE, PROGRESS_REPORT_INTERVAL,
    HIGH_FAILURE_RATE_THRESHOLD
)
from .utils import (
    normalize_audio, clean_text_for_tts, ProgressReporter,
    format_duration
)

logger = logging.getLogger(__name__)

__all__ = ['TTSGenerator', 'AudioResult']


@dataclass
class AudioResult:
    """Result of TTS generation for a sentence."""
    sentence_id: int
    audio_path: Path
    duration: float
    sample_rate: int
    success: bool
    error_message: Optional[str] = None


class TTSGenerator:
    """Generates TTS audio using Kokoro for sentence groups."""

    def __init__(self, config: Config):
        self.config = config
        self.pipeline: Optional[KPipeline] = None
        self.generation_stats = {
            'total_sentences': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_duration': 0.0,
            'total_audio_duration': 0.0,
            'average_generation_time': 0.0
        }
        self.logger = logger

    def initialize_pipeline(self) -> bool:
        """Initialize the Kokoro TTS pipeline."""
        try:
            self.logger.info(f"Initializing Kokoro pipeline with voice: {self.config.voice}")
            start_time = time.time()

            # Initialize with American English and specified voice
            self.pipeline = KPipeline(lang_code='a')

            init_time = time.time() - start_time
            self.logger.info(f"Pipeline initialized successfully in {format_duration(init_time)}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Kokoro pipeline: {e}")
            return False

    def generate_sentence_audio(
        self,
        sentence_group: SentenceGroup,
        output_path: Path,
        force_regenerate: bool = False
    ) -> AudioResult:
        """
        Generate TTS audio for a single sentence group.

        Args:
            sentence_group: The sentence to generate audio for
            output_path: Path to save the audio file
            force_regenerate: Force regeneration even if file exists

        Returns:
            AudioResult with generation details
        """
        if not self.pipeline:
            return AudioResult(
                sentence_id=sentence_group.sentence_id,
                audio_path=output_path,
                duration=0.0,
                sample_rate=self.config.sample_rate,
                success=False,
                error_message="Pipeline not initialized"
            )

        # Check if file already exists and we don't want to force regenerate
        if output_path.exists() and not force_regenerate:
            try:
                # Read existing file to get duration
                audio_data, sample_rate = sf.read(output_path)
                duration = len(audio_data) / sample_rate
                return AudioResult(
                    sentence_id=sentence_group.sentence_id,
                    audio_path=output_path,
                    duration=duration,
                    sample_rate=sample_rate,
                    success=True
                )
            except Exception as e:
                print(f"Error reading existing audio file {output_path}: {e}")
                # Continue to regenerate

        try:
            start_time = time.time()

            # Clean the text for TTS
            cleaned_text = clean_text_for_tts(sentence_group.text)

            self.logger.debug(f"Generating audio for sentence {sentence_group.sentence_id}: {cleaned_text[:50]}...")

            # Generate TTS audio
            generator = self.pipeline(cleaned_text, voice=self.config.voice)

            # Process and concatenate all chunks
            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(generator):
                if audio is not None:
                    audio_np = audio.numpy()
                    audio_chunks.append(audio_np)

                    self.logger.debug(f"  Chunk {i}: {audio_np.shape}")

            if not audio_chunks:
                return AudioResult(
                    sentence_id=sentence_group.sentence_id,
                    audio_path=output_path,
                    duration=0.0,
                    sample_rate=self.config.sample_rate,
                    success=False,
                    error_message="No audio generated"
                )

            # Concatenate all audio chunks
            final_audio = np.concatenate(audio_chunks)

            # Normalize audio to prevent clipping
            final_audio = normalize_audio(final_audio)

            # Save audio file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                output_path,
                final_audio,
                self.config.sample_rate,
                subtype=AUDIO_SUBTYPE
            )

            generation_time = time.time() - start_time
            audio_duration = len(final_audio) / self.config.sample_rate

            # Update statistics
            self.generation_stats['successful_generations'] += 1
            self.generation_stats['total_duration'] += generation_time
            self.generation_stats['total_audio_duration'] += audio_duration

            self.logger.debug(f"Generated {format_duration(audio_duration)} audio in {format_duration(generation_time)}")

            return AudioResult(
                sentence_id=sentence_group.sentence_id,
                audio_path=output_path,
                duration=audio_duration,
                sample_rate=self.config.sample_rate,
                success=True
            )

        except Exception as e:
            error_msg = f"TTS generation failed: {e}"
            self.logger.error(f"Error generating audio for sentence {sentence_group.sentence_id}: {error_msg}")

            self.generation_stats['failed_generations'] += 1

            return AudioResult(
                sentence_id=sentence_group.sentence_id,
                audio_path=output_path,
                duration=0.0,
                sample_rate=self.config.sample_rate,
                success=False,
                error_message=error_msg
            )

    def generate_all_sentences(
        self,
        sentence_groups: List[SentenceGroup],
        working_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> List[AudioResult]:
        """
        Generate TTS audio for all sentence groups.

        Args:
            sentence_groups: List of sentence groups to process
            working_dir: Working directory for output files
            progress_callback: Optional callback for progress updates

        Returns:
            List of AudioResult objects
        """
        if not self.pipeline:
            if not self.initialize_pipeline():
                self.logger.error("Failed to initialize TTS pipeline")
                return []

        results = []
        self.generation_stats['total_sentences'] = len(sentence_groups)

        with ProgressReporter("TTS generation", len(sentence_groups), "dubbing.tts") as progress:
            for i, sentence_group in enumerate(sentence_groups):
                # Get output path for this sentence
                audio_path = self.config.get_audio_snippet_path(working_dir, sentence_group.sentence_id)

                # Generate audio
                result = self.generate_sentence_audio(sentence_group, audio_path)
                results.append(result)

                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(sentence_groups), result)

                # Update progress
                progress.update()

                # Check for high failure rate
                if i > 10:  # Only check after some samples
                    failure_rate = self.generation_stats['failed_generations'] / (i + 1)
                    if failure_rate > HIGH_FAILURE_RATE_THRESHOLD:
                        self.logger.warning(f"High TTS failure rate: {failure_rate*100:.1f}%")

        self._finalize_stats()
        self._log_generation_summary()
        return results

    def _log_generation_summary(self) -> None:
        """Log a summary of TTS generation results."""
        stats = self.generation_stats
        total = stats['total_sentences']
        successful = stats['successful_generations']
        failed = stats['failed_generations']

        if total > 0:
            success_rate = (successful / total) * 100
            self.logger.info(f"TTS Generation Summary: {successful}/{total} successful ({success_rate:.1f}%)")

            if failed > 0:
                self.logger.warning(f"TTS Generation Failures: {failed} sentences failed")

            if successful > 0:
                avg_time = stats['average_generation_time']
                total_audio = stats['total_audio_duration']
                total_time = stats['total_duration']
                speed = total_audio / total_time if total_time > 0 else 0

                self.logger.info(f"Average generation time: {format_duration(avg_time)} per sentence")
                self.logger.info(f"Total audio generated: {format_duration(total_audio)}")
                self.logger.info(f"Generation speed: {speed:.2f}x real-time")

    def _finalize_stats(self) -> None:
        """Calculate final statistics after all generations."""
        if self.generation_stats['successful_generations'] > 0:
            self.generation_stats['average_generation_time'] = (
                self.generation_stats['total_duration'] /
                self.generation_stats['successful_generations']
            )

    def get_generation_stats(self) -> Dict:
        """Get statistics about TTS generation performance."""
        return self.generation_stats.copy()

    def print_stats(self) -> None:
        """Print generation statistics (deprecated - use logging instead)."""
        self.logger.warning("print_stats() is deprecated, statistics are now logged automatically")
        self._log_generation_summary()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.pipeline:
            self.logger.debug("Cleaning up TTS pipeline")
            self.pipeline = None


def main():
    """Test the TTS generator with sample data."""
    from .srt_parser import SRTParser
    from .sentence_grouper import SentenceGrouper
    from .config import get_default_config
    from .utils import setup_logging

    setup_logging("DEBUG")
    logger = logging.getLogger(__name__)

    config = get_default_config()
    parser = SRTParser()
    grouper = SentenceGrouper()
    tts_generator = TTSGenerator(config)

    try:
        # Parse and group sentences
        logger.info("Testing TTS generator")
        entries = parser.parse_file('/mnt/d/Coloso/Syagamu/01.srt')
        sentences = grouper.group_sentences(entries[:5])  # Test with first 5 entries
        working_dir = config.create_working_dir("test_tts")

        # Generate TTS for sentences
        results = tts_generator.generate_all_sentences(sentences, working_dir)

        # Log results summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Test completed: {successful}/{len(results)} successful generations")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tts_generator.cleanup()


if __name__ == "__main__":
    main()