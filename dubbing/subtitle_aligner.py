"""
Subtitle Aligner Module
Realigns subtitle timings to match actual spoken words using faster-whisper transcription.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher

from .srt_parser import SubtitleEntry
from .sentence_grouper import SentenceGroup
from .audio_assembler import TimingEvent
from .config import Config


@dataclass
class WordTimestamp:
    """Represents a word with its timing information from Whisper."""
    word: str
    start: float
    end: float
    probability: float


@dataclass
class AlignedSubtitle:
    """Represents a subtitle entry with aligned timing."""
    entry_id: int
    text: str
    original_start: float
    original_end: float
    aligned_start: float
    aligned_end: float
    confidence: float


@dataclass
class AlignmentStats:
    """Statistics about the alignment process."""
    total_subtitles: int
    successfully_aligned: int
    failed_alignments: int
    average_timing_adjustment: float
    max_timing_adjustment: float
    whisper_transcription_time: float
    alignment_processing_time: float


class SubtitleAligner:
    """Aligns subtitle timings with actual spoken words using faster-whisper."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.alignment_stats = AlignmentStats(
            total_subtitles=0,
            successfully_aligned=0,
            failed_alignments=0,
            average_timing_adjustment=0.0,
            max_timing_adjustment=0.0,
            whisper_transcription_time=0.0,
            alignment_processing_time=0.0
        )


    def align_subtitles(
        self,
        subtitle_entries: List[SubtitleEntry],
        audio_path: Path,
        output_path: Path,
        timeline_events: Optional[List[TimingEvent]] = None,
        sentence_groups: Optional[List[SentenceGroup]] = None
    ) -> bool:
        """
        Align subtitle timings with actual spoken words.

        Args:
            subtitle_entries: Original subtitle entries
            audio_path: Path to the dubbed audio file
            output_path: Path to save aligned subtitle file
            timeline_events: Timeline events from audio assembly (optional)
            sentence_groups: Sentence groups used for TTS (optional)

        Returns:
            True if alignment was successful
        """
        import time
        start_time = time.time()

        try:
            print("\n=== Phase: Subtitle Alignment ===")

            # Transcribe audio with word-level timestamps using subprocess
            word_timestamps = self._transcribe_with_timestamps(audio_path)
            if not word_timestamps:
                print("Failed to transcribe audio")
                return False

            self.alignment_stats.whisper_transcription_time = time.time() - start_time

            # Align subtitles with word timestamps
            alignment_start = time.time()
            aligned_subtitles = self._align_subtitle_timings(
                subtitle_entries, word_timestamps, timeline_events, sentence_groups
            )

            if not aligned_subtitles:
                print("Failed to align subtitle timings")
                return False

            self.alignment_stats.alignment_processing_time = time.time() - alignment_start

            # Generate aligned SRT file
            success = self._generate_aligned_srt(aligned_subtitles, output_path)

            if success:
                self._calculate_alignment_stats(aligned_subtitles)
                self._save_alignment_debug_info(
                    aligned_subtitles, word_timestamps,
                    output_path.parent / "alignment_debug.json"
                )
                print(f"Aligned subtitles saved: {output_path}")
                self._print_alignment_stats()

            return success

        except Exception as e:
            print(f"Error during subtitle alignment: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _transcribe_with_timestamps(self, audio_path: Path) -> List[WordTimestamp]:
        """Transcribe audio and extract word-level timestamps."""
        try:
            print(f"Transcribing audio with Whisper: {audio_path}")

            # Transcribe with word-level timestamps
            segments, info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=True,
                language="en"
            )

            print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

            word_timestamps = []

            for segment in segments:
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_timestamps.append(WordTimestamp(
                            word=word.word.strip(),
                            start=word.start,
                            end=word.end,
                            probability=getattr(word, 'probability', 1.0)
                        ))

            print(f"Extracted {len(word_timestamps)} word timestamps")

            # Debug: Print first few words
            if word_timestamps and self.config.verbose_logging:
                print("Sample word timestamps:")
                for i, wt in enumerate(word_timestamps[:10]):
                    print(f"  {i+1}: '{wt.word}' ({wt.start:.3f}s - {wt.end:.3f}s)")

            return word_timestamps

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return []

    def _align_subtitle_timings(
        self,
        subtitle_entries: List[SubtitleEntry],
        word_timestamps: List[WordTimestamp],
        timeline_events: Optional[List[TimingEvent]] = None,
        sentence_groups: Optional[List[SentenceGroup]] = None
    ) -> List[AlignedSubtitle]:
        """Align subtitle entries with word timestamps."""
        try:
            print(f"Aligning {len(subtitle_entries)} subtitle entries with word timestamps")

            aligned_subtitles = []

            # Create full transcription text for matching
            transcription_text = " ".join([wt.word for wt in word_timestamps]).lower()

            for entry in subtitle_entries:
                aligned_subtitle = self._align_single_subtitle(
                    entry, word_timestamps, transcription_text
                )

                if aligned_subtitle:
                    aligned_subtitles.append(aligned_subtitle)
                    if self.config.verbose_logging:
                        timing_diff = aligned_subtitle.aligned_start - aligned_subtitle.original_start
                        print(f"  Entry {entry.id}: {timing_diff:+.3f}s adjustment")

            return aligned_subtitles

        except Exception as e:
            print(f"Error aligning subtitle timings: {e}")
            return []

    def _align_single_subtitle(
        self,
        entry: SubtitleEntry,
        word_timestamps: List[WordTimestamp],
        transcription_text: str
    ) -> Optional[AlignedSubtitle]:
        """Align a single subtitle entry with word timestamps."""
        try:
            # Clean subtitle text for matching
            subtitle_text = self._clean_text_for_matching(entry.text)

            if not subtitle_text.strip():
                return None

            # Find the best match in transcription
            match_start, match_end, confidence = self._find_text_match_in_transcription(
                subtitle_text, word_timestamps, transcription_text
            )

            if match_start is None or match_end is None:
                # Fallback: use original timing
                return AlignedSubtitle(
                    entry_id=entry.id,
                    text=entry.text,
                    original_start=entry.start_time,
                    original_end=entry.end_time,
                    aligned_start=entry.start_time,
                    aligned_end=entry.end_time,
                    confidence=0.0
                )

            # Add small padding for readability
            padding = getattr(self.config, 'alignment_padding', 0.05)
            aligned_start = max(0, match_start - padding)
            aligned_end = match_end + padding

            return AlignedSubtitle(
                entry_id=entry.id,
                text=entry.text,
                original_start=entry.start_time,
                original_end=entry.end_time,
                aligned_start=aligned_start,
                aligned_end=aligned_end,
                confidence=confidence
            )

        except Exception as e:
            print(f"Error aligning subtitle entry {entry.id}: {e}")
            return None

    def _find_text_match_in_transcription(
        self,
        subtitle_text: str,
        word_timestamps: List[WordTimestamp],
        transcription_text: str
    ) -> Tuple[Optional[float], Optional[float], float]:
        """Find where subtitle text appears in the transcribed word timestamps."""
        try:
            # Split subtitle into words for matching
            subtitle_words = subtitle_text.split()

            if not subtitle_words:
                return None, None, 0.0

            # Find best sequence match in word timestamps
            best_match_start = None
            best_match_end = None
            best_score = 0.0

            # Try to find consecutive word matches
            for i in range(len(word_timestamps) - len(subtitle_words) + 1):
                # Extract sequence of words from timestamps
                word_sequence = []
                for j in range(len(subtitle_words)):
                    if i + j < len(word_timestamps):
                        word_sequence.append(word_timestamps[i + j].word.lower().strip())

                # Calculate similarity score
                transcribed_phrase = " ".join(word_sequence)
                similarity = SequenceMatcher(None, subtitle_text, transcribed_phrase).ratio()

                if similarity > best_score and similarity > 0.5:  # Minimum 50% similarity
                    best_score = similarity
                    best_match_start = word_timestamps[i].start
                    best_match_end = word_timestamps[min(i + len(subtitle_words) - 1, len(word_timestamps) - 1)].end

            # If no good match found, try fuzzy matching individual words
            if best_match_start is None:
                first_word = subtitle_words[0].lower()
                last_word = subtitle_words[-1].lower()

                first_match_idx = None
                last_match_idx = None

                # Find first word
                for i, wt in enumerate(word_timestamps):
                    if self._words_similar(first_word, wt.word.lower().strip()):
                        first_match_idx = i
                        break

                # Find last word (searching from first match onwards)
                if first_match_idx is not None:
                    for i in range(first_match_idx, min(first_match_idx + len(subtitle_words) * 2, len(word_timestamps))):
                        if self._words_similar(last_word, word_timestamps[i].word.lower().strip()):
                            last_match_idx = i

                if first_match_idx is not None and last_match_idx is not None:
                    best_match_start = word_timestamps[first_match_idx].start
                    best_match_end = word_timestamps[last_match_idx].end
                    best_score = 0.3  # Lower confidence for fuzzy match

            return best_match_start, best_match_end, best_score

        except Exception as e:
            print(f"Error finding text match: {e}")
            return None, None, 0.0

    def _words_similar(self, word1: str, word2: str, threshold: float = 0.8) -> bool:
        """Check if two words are similar enough to be considered a match."""
        if word1 == word2:
            return True

        # Remove common punctuation
        word1 = re.sub(r'[^\w]', '', word1).lower()
        word2 = re.sub(r'[^\w]', '', word2).lower()

        if len(word1) < 3 or len(word2) < 3:
            return word1 == word2

        similarity = SequenceMatcher(None, word1, word2).ratio()
        return similarity >= threshold

    def _clean_text_for_matching(self, text: str) -> str:
        """Clean text for better matching with transcription."""
        # Convert to lowercase
        text = text.lower()

        # Remove common punctuation but keep apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _generate_aligned_srt(self, aligned_subtitles: List[AlignedSubtitle], output_path: Path) -> bool:
        """Generate SRT file with aligned timings."""
        try:
            def format_time(seconds: float) -> str:
                """Convert seconds to SRT time format."""
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                milliseconds = int((seconds % 1) * 1000)
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

            # Sort by aligned start time
            aligned_subtitles.sort(key=lambda x: x.aligned_start)

            # Ensure no overlapping subtitles
            adjusted_subtitles = []
            for i, subtitle in enumerate(aligned_subtitles):
                start_time = subtitle.aligned_start
                end_time = subtitle.aligned_end

                # Check for overlap with next subtitle
                if i + 1 < len(aligned_subtitles):
                    next_start = aligned_subtitles[i + 1].aligned_start
                    if end_time > next_start:
                        # Adjust end time to avoid overlap
                        end_time = max(start_time + 0.1, next_start - 0.05)

                adjusted_subtitles.append(AlignedSubtitle(
                    entry_id=subtitle.entry_id,
                    text=subtitle.text,
                    original_start=subtitle.original_start,
                    original_end=subtitle.original_end,
                    aligned_start=start_time,
                    aligned_end=end_time,
                    confidence=subtitle.confidence
                ))

            # Write SRT file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, subtitle in enumerate(adjusted_subtitles, 1):
                    start_time_str = format_time(subtitle.aligned_start)
                    end_time_str = format_time(subtitle.aligned_end)

                    f.write(f"{i}\n")
                    f.write(f"{start_time_str} --> {end_time_str}\n")
                    f.write(f"{subtitle.text}\n\n")

            return True

        except Exception as e:
            print(f"Error generating aligned SRT file: {e}")
            return False

    def _calculate_alignment_stats(self, aligned_subtitles: List[AlignedSubtitle]) -> None:
        """Calculate alignment statistics."""
        self.alignment_stats.total_subtitles = len(aligned_subtitles)

        timing_adjustments = []
        successful_alignments = 0

        for subtitle in aligned_subtitles:
            adjustment = abs(subtitle.aligned_start - subtitle.original_start)
            timing_adjustments.append(adjustment)

            if subtitle.confidence > 0.3:  # Consider successful if confidence > 30%
                successful_alignments += 1

        self.alignment_stats.successfully_aligned = successful_alignments
        self.alignment_stats.failed_alignments = len(aligned_subtitles) - successful_alignments

        if timing_adjustments:
            self.alignment_stats.average_timing_adjustment = sum(timing_adjustments) / len(timing_adjustments)
            self.alignment_stats.max_timing_adjustment = max(timing_adjustments)

    def _save_alignment_debug_info(
        self,
        aligned_subtitles: List[AlignedSubtitle],
        word_timestamps: List[WordTimestamp],
        debug_path: Path
    ) -> None:
        """Save alignment debugging information."""
        try:
            debug_data = {
                'alignment_stats': asdict(self.alignment_stats),
                'aligned_subtitles': [asdict(sub) for sub in aligned_subtitles],
                'word_timestamps_sample': [asdict(wt) for wt in word_timestamps[:50]],  # First 50 words
                'config': {
                    'whisper_model_size': getattr(self.config, 'whisper_model_size', 'large-v3'),
                    'alignment_padding': getattr(self.config, 'alignment_padding', 0.05),
                    'align_subtitles': getattr(self.config, 'align_subtitles', True)
                }
            }

            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2)

            print(f"Alignment debug info saved: {debug_path}")

        except Exception as e:
            print(f"Error saving alignment debug info: {e}")

    def _print_alignment_stats(self) -> None:
        """Print alignment statistics."""
        stats = self.alignment_stats

        print(f"\nSubtitle Alignment Statistics:")
        print(f"  Total subtitles: {stats.total_subtitles}")
        print(f"  Successfully aligned: {stats.successfully_aligned}")
        print(f"  Failed alignments: {stats.failed_alignments}")
        print(f"  Success rate: {(stats.successfully_aligned / stats.total_subtitles * 100):.1f}%")
        print(f"  Average timing adjustment: {stats.average_timing_adjustment:.3f}s")
        print(f"  Maximum timing adjustment: {stats.max_timing_adjustment:.3f}s")
        print(f"  Whisper transcription time: {stats.whisper_transcription_time:.1f}s")
        print(f"  Alignment processing time: {stats.alignment_processing_time:.1f}s")

    def get_alignment_stats(self) -> AlignmentStats:
        """Get alignment statistics."""
        return self.alignment_stats


def main():
    """Test subtitle alignment functionality."""
    from .config import get_default_config
    from .srt_parser import SRTParser

    config = get_default_config()
    config.verbose_logging = True

    # Add alignment settings
    config.whisper_model_size = 'large-v3'
    config.alignment_padding = 0.05
    config.align_subtitles = True

    aligner = SubtitleAligner(config)
    parser = SRTParser()

    try:
        # Test with actual files
        working_dir = Path("/mnt/d/workspace/kokoro/working/01_Before_Starting_the_Class")

        subtitle_entries = parser.parse_file(str(working_dir / "subtitles.srt"))
        audio_path = working_dir / "final_audio.wav"
        output_path = working_dir / "subtitles_aligned.srt"

        print(f"Testing alignment with {len(subtitle_entries)} subtitles")

        success = aligner.align_subtitles(
            subtitle_entries, audio_path, output_path
        )

        if success:
            print("Alignment test completed successfully!")
        else:
            print("Alignment test failed!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()