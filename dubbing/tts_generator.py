"""
TTS Generator Module
Generates audio snippets using Kokoro TTS for sentence groups.
"""

import os
import time
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from kokoro import KPipeline
from .sentence_grouper import SentenceGroup
from .config import Config


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

    def initialize_pipeline(self) -> bool:
        """Initialize the Kokoro TTS pipeline."""
        try:
            print(f"Initializing Kokoro pipeline with voice: {self.config.voice}")
            start_time = time.time()

            # Initialize with American English and specified voice
            self.pipeline = KPipeline(lang_code='a')

            init_time = time.time() - start_time
            print(f"Pipeline initialized successfully in {init_time:.2f}s")
            return True

        except Exception as e:
            print(f"Failed to initialize Kokoro pipeline: {e}")
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
            cleaned_text = self._clean_text_for_tts(sentence_group.text)

            if self.config.verbose_logging:
                print(f"Generating audio for sentence {sentence_group.sentence_id}: {cleaned_text[:50]}...")

            # Generate TTS audio
            generator = self.pipeline(cleaned_text, voice=self.config.voice)

            # Process and concatenate all chunks
            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(generator):
                if audio is not None:
                    audio_np = audio.numpy()
                    audio_chunks.append(audio_np)

                    if self.config.verbose_logging:
                        print(f"  Chunk {i}: {audio_np.shape}")

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
            final_audio = self._normalize_audio(final_audio)

            # Save audio file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                output_path,
                final_audio,
                self.config.sample_rate,
                subtype='PCM_16'
            )

            generation_time = time.time() - start_time
            audio_duration = len(final_audio) / self.config.sample_rate

            # Update statistics
            self.generation_stats['successful_generations'] += 1
            self.generation_stats['total_duration'] += generation_time
            self.generation_stats['total_audio_duration'] += audio_duration

            if self.config.verbose_logging:
                print(f"  Generated {audio_duration:.2f}s audio in {generation_time:.2f}s")

            return AudioResult(
                sentence_id=sentence_group.sentence_id,
                audio_path=output_path,
                duration=audio_duration,
                sample_rate=self.config.sample_rate,
                success=True
            )

        except Exception as e:
            error_msg = f"TTS generation failed: {e}"
            print(f"Error generating audio for sentence {sentence_group.sentence_id}: {error_msg}")

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
        progress_callback: Optional[callable] = None
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
                return []

        results = []
        self.generation_stats['total_sentences'] = len(sentence_groups)

        print(f"Generating TTS audio for {len(sentence_groups)} sentences...")

        for i, sentence_group in enumerate(sentence_groups):
            # Get output path for this sentence
            audio_path = self.config.get_audio_snippet_path(working_dir, sentence_group.sentence_id)

            # Generate audio
            result = self.generate_sentence_audio(sentence_group, audio_path)
            results.append(result)

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(sentence_groups), result)

            # Simple progress indicator
            if (i + 1) % 10 == 0 or i == len(sentence_groups) - 1:
                success_rate = self.generation_stats['successful_generations'] / (i + 1) * 100
                print(f"Progress: {i + 1}/{len(sentence_groups)} ({success_rate:.1f}% success)")

        self._finalize_stats()
        return results

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for optimal TTS generation."""
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Handle common text-to-speech issues
        replacements = {
            '&': 'and',
            '@': 'at',
            '%': 'percent',
            '#': 'hashtag',
            '$': 'dollar',
            '€': 'euro',
            '£': 'pound',
            '¥': 'yen',
            '©': 'copyright',
            '®': 'registered',
            '™': 'trademark',
            '...': '. . .',  # Make ellipsis more pronounceable
            '--': ', ',      # Replace dashes with pauses
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'

        return text

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping and ensure proper range."""
        if len(audio) == 0:
            return audio

        # Find peak amplitude
        max_amplitude = max(abs(audio.max()), abs(audio.min()))

        if max_amplitude > 1.0:
            # Normalize to prevent clipping
            audio = audio / max_amplitude
        elif max_amplitude < 0.1:
            # Boost very quiet audio (but be conservative)
            audio = audio * (0.5 / max_amplitude)

        # Apply soft limiting to prevent any remaining issues
        audio = np.tanh(audio * 0.9) * 0.95

        return audio

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
        """Print generation statistics."""
        stats = self.generation_stats
        total = stats['total_sentences']
        successful = stats['successful_generations']
        failed = stats['failed_generations']

        print(f"\nTTS Generation Statistics:")
        print(f"  Total sentences: {total}")
        print(f"  Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"  Failed: {failed} ({failed/total*100:.1f}%)")
        if successful > 0:
            print(f"  Average generation time: {stats['average_generation_time']:.2f}s per sentence")
            print(f"  Total audio generated: {stats['total_audio_duration']:.1f}s")
            print(f"  Generation speed: {stats['total_audio_duration']/stats['total_duration']:.2f}x real-time")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.pipeline = None


def main():
    """Test the TTS generator with sample data."""
    from .srt_parser import SRTParser
    from .sentence_grouper import SentenceGrouper
    from .config import get_default_config

    config = get_default_config()
    config.verbose_logging = True

    parser = SRTParser()
    grouper = SentenceGrouper()
    tts_generator = TTSGenerator(config)

    try:
        # Parse and group sentences
        print("Parsing SRT file...")
        entries = parser.parse_file('/mnt/d/Coloso/Syagamu/01.srt')
        print(f"Parsed {len(entries)} entries")

        print("Grouping sentences...")
        sentences = grouper.group_sentences(entries[:10])  # Test with first 10 entries
        print(f"Created {len(sentences)} sentence groups")

        # Create working directory
        working_dir = config.create_working_dir("test_tts")

        # Generate TTS for all sentences
        results = tts_generator.generate_all_sentences(sentences, working_dir)

        # Show results
        print(f"\nGenerated audio for {len(results)} sentences")
        for result in results:
            status = "SUCCESS" if result.success else f"FAILED: {result.error_message}"
            print(f"Sentence {result.sentence_id}: {status}")
            if result.success:
                print(f"  Duration: {result.duration:.2f}s")
                print(f"  File: {result.audio_path}")

        tts_generator.print_stats()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tts_generator.cleanup()


if __name__ == "__main__":
    main()