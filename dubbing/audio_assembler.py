"""
Audio Assembler Module
Assembles individual sentence audio snippets into a complete audio track with intelligent timing.
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

from .sentence_grouper import SentenceGroup
from .tts_generator import AudioResult
from .config import Config


@dataclass
class TimingEvent:
    """Represents a timed event in the final audio timeline."""
    event_type: str           # 'silence', 'audio', 'sentence'
    start_time: float         # Start time in seconds
    duration: float           # Duration in seconds
    sentence_id: Optional[int] = None
    audio_path: Optional[Path] = None
    ideal_start_time: Optional[float] = None  # Original subtitle timing
    timing_deviation: Optional[float] = None  # Actual vs ideal start time


@dataclass
class AssemblyStats:
    """Statistics about the audio assembly process."""
    total_sentences: int
    total_duration: float
    total_audio_duration: float
    total_silence_duration: float
    average_timing_deviation: float
    max_timing_deviation: float
    sentences_on_time: int
    sentences_with_gap: int
    sentences_overflowing: int


class AudioAssembler:
    """Assembles sentence audio snippets into a complete timeline."""

    def __init__(self, config: Config):
        self.config = config
        self.timeline: List[TimingEvent] = []
        self.assembly_stats = AssemblyStats(
            total_sentences=0,
            total_duration=0.0,
            total_audio_duration=0.0,
            total_silence_duration=0.0,
            average_timing_deviation=0.0,
            max_timing_deviation=0.0,
            sentences_on_time=0,
            sentences_with_gap=0,
            sentences_overflowing=0
        )

    def assemble_audio(
        self,
        sentence_groups: List[SentenceGroup],
        audio_results: List[AudioResult],
        output_path: Path,
        save_timeline_debug: bool = True
    ) -> bool:
        """
        Assemble individual audio snippets into a complete audio track.

        Args:
            sentence_groups: List of sentence groups with timing information
            audio_results: List of TTS generation results
            output_path: Path for the final audio file
            save_timeline_debug: Whether to save timeline debugging info

        Returns:
            True if assembly was successful
        """
        try:
            # Create mapping from sentence_id to audio result
            audio_map = {result.sentence_id: result for result in audio_results}

            # Build timeline
            self._build_timeline(sentence_groups, audio_map)

            # Generate the final audio
            success = self._generate_final_audio(output_path)

            if success and save_timeline_debug:
                self._save_timeline_debug(output_path.parent / "timeline_debug.json")

            return success

        except Exception as e:
            print(f"Error assembling audio: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _build_timeline(
        self,
        sentence_groups: List[SentenceGroup],
        audio_map: Dict[int, AudioResult]
    ) -> None:
        """Build the complete audio timeline with intelligent timing."""
        self.timeline = []
        current_time = 0.0
        timing_deviations = []

        print(f"Building timeline for {len(sentence_groups)} sentences...")

        for i, sentence_group in enumerate(sentence_groups):
            sentence_id = sentence_group.sentence_id
            ideal_start = sentence_group.start_time

            # Get audio result for this sentence
            if sentence_id not in audio_map or not audio_map[sentence_id].success:
                print(f"Warning: No audio available for sentence {sentence_id}, skipping")
                continue

            audio_result = audio_map[sentence_id]
            audio_duration = audio_result.duration

            # Determine actual placement time
            if current_time <= ideal_start:
                # We can start on time or early
                if current_time < ideal_start:
                    # Insert silence to catch up to ideal start time
                    silence_duration = ideal_start - current_time
                    self.timeline.append(TimingEvent(
                        event_type='silence',
                        start_time=current_time,
                        duration=silence_duration
                    ))
                    self.assembly_stats.total_silence_duration += silence_duration

                actual_start = ideal_start
                self.assembly_stats.sentences_on_time += 1

            else:
                # We're running late - add minimum gap
                gap_duration = self.config.min_sentence_gap
                if gap_duration > 0:
                    self.timeline.append(TimingEvent(
                        event_type='silence',
                        start_time=current_time,
                        duration=gap_duration
                    ))
                    self.assembly_stats.total_silence_duration += gap_duration
                    current_time += gap_duration

                actual_start = current_time
                if actual_start > ideal_start:
                    self.assembly_stats.sentences_overflowing += 1
                else:
                    self.assembly_stats.sentences_with_gap += 1

            # Calculate timing deviation
            deviation = actual_start - ideal_start
            timing_deviations.append(abs(deviation))

            # Add the sentence audio to timeline
            self.timeline.append(TimingEvent(
                event_type='sentence',
                start_time=actual_start,
                duration=audio_duration,
                sentence_id=sentence_id,
                audio_path=audio_result.audio_path,
                ideal_start_time=ideal_start,
                timing_deviation=deviation
            ))

            # Update current time
            current_time = actual_start + audio_duration
            self.assembly_stats.total_audio_duration += audio_duration

            if self.config.verbose_logging:
                status = "ON_TIME" if deviation == 0 else f"LATE+{deviation:.3f}s" if deviation > 0 else f"EARLY{deviation:.3f}s"
                print(f"  Sentence {sentence_id}: {actual_start:.3f}s ({status}) -> {current_time:.3f}s")

        # Calculate final statistics
        self.assembly_stats.total_sentences = len([e for e in self.timeline if e.event_type == 'sentence'])
        self.assembly_stats.total_duration = current_time
        if timing_deviations:
            self.assembly_stats.average_timing_deviation = sum(timing_deviations) / len(timing_deviations)
            self.assembly_stats.max_timing_deviation = max(timing_deviations)

        print(f"Timeline built: {self.assembly_stats.total_duration:.2f}s total, {self.assembly_stats.total_sentences} sentences")

    def _generate_final_audio(self, output_path: Path) -> bool:
        """Generate the final audio file from the timeline."""
        if not self.timeline:
            print("Error: No timeline events to process")
            return False

        try:
            print(f"Generating final audio file: {output_path}")

            # Calculate total samples needed
            total_duration = self.assembly_stats.total_duration
            total_samples = int(total_duration * self.config.sample_rate)

            # Initialize output audio array
            final_audio = np.zeros(total_samples, dtype=np.float32)

            for event in self.timeline:
                start_sample = int(event.start_time * self.config.sample_rate)
                end_sample = int((event.start_time + event.duration) * self.config.sample_rate)

                if event.event_type == 'silence':
                    # Silence is already zeros, nothing to do
                    continue

                elif event.event_type == 'sentence' and event.audio_path:
                    # Load and place audio
                    try:
                        audio_data, sample_rate = sf.read(event.audio_path)

                        # Resample if necessary
                        if sample_rate != self.config.sample_rate:
                            print(f"Warning: Resampling audio from {sample_rate}Hz to {self.config.sample_rate}Hz")
                            audio_data = self._resample_audio(audio_data, sample_rate, self.config.sample_rate)

                        # Ensure we don't exceed array bounds
                        audio_samples = len(audio_data)
                        available_samples = total_samples - start_sample
                        samples_to_copy = min(audio_samples, available_samples)

                        if samples_to_copy > 0:
                            final_audio[start_sample:start_sample + samples_to_copy] = audio_data[:samples_to_copy]

                        if self.config.verbose_logging and samples_to_copy < audio_samples:
                            print(f"Warning: Truncated audio for sentence {event.sentence_id} ({audio_samples - samples_to_copy} samples)")

                    except Exception as e:
                        print(f"Error loading audio for sentence {event.sentence_id}: {e}")

            # Normalize final audio
            final_audio = self._normalize_final_audio(final_audio)

            # Save final audio file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                output_path,
                final_audio,
                self.config.sample_rate,
                subtype='PCM_16'
            )

            print(f"Final audio saved: {output_path}")
            print(f"Duration: {len(final_audio) / self.config.sample_rate:.2f}s")

            return True

        except Exception as e:
            print(f"Error generating final audio: {e}")
            return False

    def _resample_audio(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple resampling (basic linear interpolation)."""
        if from_rate == to_rate:
            return audio

        # Calculate resampling ratio
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)

        # Simple linear interpolation resampling
        old_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(old_indices, np.arange(len(audio)), audio)

        return resampled.astype(np.float32)

    def _normalize_final_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize the final audio to optimal levels."""
        if len(audio) == 0:
            return audio

        # Find peak amplitude
        max_amplitude = max(abs(audio.max()), abs(audio.min()))

        if max_amplitude > 0.95:
            # Normalize to prevent clipping
            audio = audio * (0.95 / max_amplitude)
        elif max_amplitude < 0.1:
            # Boost very quiet audio
            audio = audio * (0.7 / max_amplitude)

        return audio

    def _save_timeline_debug(self, debug_path: Path) -> None:
        """Save timeline debugging information."""
        debug_data = {
            'timeline_events': [asdict(event) for event in self.timeline],
            'assembly_stats': asdict(self.assembly_stats),
            'config': {
                'sample_rate': self.config.sample_rate,
                'min_sentence_gap': self.config.min_sentence_gap,
                'voice': self.config.voice
            }
        }

        # Convert Path objects to strings for JSON serialization
        for event in debug_data['timeline_events']:
            if event.get('audio_path'):
                event['audio_path'] = str(event['audio_path'])

        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2)

        print(f"Timeline debug info saved: {debug_path}")

    def get_assembly_stats(self) -> AssemblyStats:
        """Get assembly statistics."""
        return self.assembly_stats

    def print_stats(self) -> None:
        """Print assembly statistics."""
        stats = self.assembly_stats

        print(f"\nAudio Assembly Statistics:")
        print(f"  Total sentences: {stats.total_sentences}")
        print(f"  Total duration: {stats.total_duration:.2f}s")
        print(f"  Audio duration: {stats.total_audio_duration:.2f}s")
        print(f"  Silence duration: {stats.total_silence_duration:.2f}s")
        print(f"  Audio/Total ratio: {stats.total_audio_duration/stats.total_duration*100:.1f}%")

        if stats.total_sentences > 0:
            print(f"\nTiming Statistics:")
            print(f"  Sentences on time: {stats.sentences_on_time}")
            print(f"  Sentences with gap: {stats.sentences_with_gap}")
            print(f"  Sentences overflowing: {stats.sentences_overflowing}")
            print(f"  Average timing deviation: {stats.average_timing_deviation:.3f}s")
            print(f"  Max timing deviation: {stats.max_timing_deviation:.3f}s")

    def analyze_timing(self) -> Dict:
        """Analyze timing patterns and return insights."""
        sentence_events = [e for e in self.timeline if e.event_type == 'sentence']

        if not sentence_events:
            return {'error': 'No sentence events in timeline'}

        # Calculate timing analysis
        deviations = [abs(e.timing_deviation) for e in sentence_events if e.timing_deviation is not None]
        gaps = [e.duration for e in self.timeline if e.event_type == 'silence']

        analysis = {
            'sentence_count': len(sentence_events),
            'timing_accuracy': {
                'perfect_timing': len([d for d in deviations if d == 0]),
                'good_timing': len([d for d in deviations if 0 < d <= 0.5]),
                'poor_timing': len([d for d in deviations if d > 1.0]),
            },
            'silence_analysis': {
                'total_gaps': len(gaps),
                'total_silence': sum(gaps),
                'average_gap': sum(gaps) / len(gaps) if gaps else 0,
                'longest_gap': max(gaps) if gaps else 0,
            },
            'recommendations': []
        }

        # Generate recommendations
        if analysis['timing_accuracy']['poor_timing'] > len(sentence_events) * 0.2:
            analysis['recommendations'].append("Consider reducing minimum sentence gap for better timing")

        if analysis['silence_analysis']['total_silence'] > self.assembly_stats.total_duration * 0.3:
            analysis['recommendations'].append("High silence ratio - original timing may be too sparse")

        return analysis


def main():
    """Test the audio assembler with sample data."""
    from .srt_parser import SRTParser
    from .sentence_grouper import SentenceGrouper
    from .tts_generator import TTSGenerator
    from .config import get_default_config

    config = get_default_config()
    config.verbose_logging = True

    parser = SRTParser()
    grouper = SentenceGrouper()
    tts_generator = TTSGenerator(config)
    assembler = AudioAssembler(config)

    try:
        # Parse and group sentences
        print("Parsing SRT file...")
        entries = parser.parse_file('/mnt/d/Coloso/Syagamu/01.srt')
        sentences = grouper.group_sentences(entries[:20])  # Test with first 20 entries

        # Create working directory
        working_dir = config.create_working_dir("test_assembly")

        # Generate TTS audio
        print("Generating TTS audio...")
        audio_results = tts_generator.generate_all_sentences(sentences, working_dir)

        # Assemble final audio
        print("Assembling final audio...")
        final_audio_path = config.get_final_audio_path(working_dir)
        success = assembler.assemble_audio(sentences, audio_results, final_audio_path)

        if success:
            assembler.print_stats()

            # Analyze timing
            timing_analysis = assembler.analyze_timing()
            print(f"\nTiming Analysis:")
            print(f"  Perfect timing: {timing_analysis['timing_accuracy']['perfect_timing']} sentences")
            print(f"  Total silence: {timing_analysis['silence_analysis']['total_silence']:.2f}s")

            if timing_analysis['recommendations']:
                print("  Recommendations:")
                for rec in timing_analysis['recommendations']:
                    print(f"    - {rec}")

        else:
            print("Assembly failed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tts_generator.cleanup()


if __name__ == "__main__":
    main()