"""
Sentence Grouper Module
Groups consecutive subtitle entries that form complete sentences.
"""

import re
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path

from .srt_parser import SubtitleEntry


@dataclass
class SentenceGroup:
    """Represents a group of subtitle entries that form a complete sentence."""
    sentence_id: int
    text: str                           # Complete sentence text
    start_time: float                   # Start time from first subtitle entry
    end_time: float                     # End time from last subtitle entry
    subtitle_entries: List[int]         # Original subtitle entry IDs
    duration_estimate: float            # end_time - start_time
    entry_count: int                    # Number of subtitle entries in this sentence


class SentenceGrouper:
    """Groups subtitle entries into complete sentences."""

    def __init__(self):
        # Common abbreviations that don't end sentences
        self.abbreviations = [
            'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Lt.', 'Sgt.', 'Col.',
            'Ltd.', 'Inc.', 'Corp.', 'Co.', 'etc.', 'vs.', 'v.', 'vs',
            'i.e.', 'e.g.', 'a.m.', 'p.m.', 'A.M.', 'P.M.',
            'St.', 'Ave.', 'Blvd.', 'Rd.', 'Jr.', 'Sr.'
        ]

        # Sentence ending punctuation
        self.sentence_endings = ['.', '!', '?']

        # Patterns that suggest continuation
        self.continuation_patterns = [
            r'^and\s+',           # Starts with "and"
            r'^but\s+',           # Starts with "but"
            r'^or\s+',            # Starts with "or"
            r'^so\s+',            # Starts with "so"
            r'^then\s+',          # Starts with "then"
            r'^however\s+',       # Starts with "however"
            r'^therefore\s+',     # Starts with "therefore"
            r'^\w+ing\s+',        # Starts with gerund (e.g., "doing", "working")
            r'^\w+ed\s+',         # Starts with past participle
        ]

        # Patterns that suggest new sentence
        self.new_sentence_patterns = [
            r'^[A-Z][a-z]',       # Starts with capital letter followed by lowercase
            r'^I\s+',             # Starts with "I "
            r'^We\s+',            # Starts with "We "
            r'^They\s+',          # Starts with "They "
            r'^This\s+',          # Starts with "This "
            r'^That\s+',          # Starts with "That "
            r'^The\s+',           # Starts with "The "
            r'^Now\s+',           # Starts with "Now "
            r'^Today\s+',         # Starts with "Today "
            r'^Next\s+',          # Starts with "Next "
        ]

    def group_sentences(
        self,
        subtitle_entries: List[SubtitleEntry],
        save_debug_info: bool = False,
        debug_path: Optional[str] = None
    ) -> List[SentenceGroup]:
        """
        Group consecutive subtitle entries that form complete sentences.

        Args:
            subtitle_entries: List of parsed subtitle entries
            save_debug_info: Whether to save debugging information
            debug_path: Path to save debug information

        Returns:
            List of SentenceGroup objects
        """
        if not subtitle_entries:
            return []

        sentence_groups = []
        current_sentence_entries = []
        sentence_id = 1

        for entry in subtitle_entries:
            current_sentence_entries.append(entry)

            # Check if this entry completes a sentence
            if self._is_sentence_complete(entry, current_sentence_entries):
                # Create sentence group
                sentence_group = self._create_sentence_group(
                    current_sentence_entries, sentence_id
                )
                sentence_groups.append(sentence_group)

                # Reset for next sentence
                current_sentence_entries = []
                sentence_id += 1

        # Handle any remaining entries that don't end with proper punctuation
        if current_sentence_entries:
            sentence_group = self._create_sentence_group(
                current_sentence_entries, sentence_id
            )
            sentence_groups.append(sentence_group)

        # Save debug information if requested
        if save_debug_info and debug_path:
            self._save_debug_info(sentence_groups, debug_path)

        return sentence_groups

    def _is_sentence_complete(
        self,
        current_entry: SubtitleEntry,
        accumulated_entries: List[SubtitleEntry]
    ) -> bool:
        """
        Determine if the current entry completes a sentence.

        Args:
            current_entry: The current subtitle entry
            accumulated_entries: All entries accumulated so far for this sentence

        Returns:
            True if sentence is complete, False otherwise
        """
        text = current_entry.text.strip()

        # Check if ends with sentence-ending punctuation
        if not any(text.endswith(ending) for ending in self.sentence_endings):
            return False

        # Check if it's just an abbreviation
        if self._is_abbreviation(text):
            return False

        # Check if the next entry (if available) suggests continuation
        # Note: We don't have access to the next entry here, so we use other heuristics

        # Additional heuristics for sentence completion
        return self._has_complete_thought(accumulated_entries)

    def _is_abbreviation(self, text: str) -> bool:
        """Check if the text ends with a common abbreviation."""
        text = text.strip()

        # Check against known abbreviations
        for abbr in self.abbreviations:
            if text.endswith(abbr):
                return True

        # Check for single letter abbreviations (e.g., "A.", "B.")
        if re.match(r'.*\b[A-Z]\.$', text):
            return True

        return False

    def _has_complete_thought(self, entries: List[SubtitleEntry]) -> bool:
        """
        Heuristic to determine if the accumulated entries form a complete thought.

        Args:
            entries: List of subtitle entries forming potential sentence

        Returns:
            True if appears to be a complete thought
        """
        if not entries:
            return False

        # Combine all text
        full_text = ' '.join(entry.text for entry in entries)

        # Very short phrases are unlikely to be complete sentences
        if len(full_text.split()) < 3:
            return False

        # Check for subject-verb patterns (basic heuristic)
        # This is a simple check - more sophisticated NLP could be added
        words = full_text.split()

        # Look for common sentence starters that suggest completeness
        if any(full_text.strip().startswith(pattern.replace(r'^\w+', '').strip())
               for pattern in self.new_sentence_patterns):
            return True

        # Look for verb patterns that suggest complete thoughts
        verb_patterns = [
            r'\b(am|is|are|was|were|have|has|had|will|would|can|could|should|must)\b',
            r'\b\w+ed\b',  # Past tense verbs
            r'\b\w+ing\b', # Present participles
        ]

        for pattern in verb_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                return True

        return True  # Default to treating as complete if we get this far

    def _create_sentence_group(
        self,
        entries: List[SubtitleEntry],
        sentence_id: int
    ) -> SentenceGroup:
        """Create a SentenceGroup from a list of subtitle entries."""
        if not entries:
            raise ValueError("Cannot create sentence group from empty entries")

        # Combine all text with spaces
        combined_text = ' '.join(entry.text for entry in entries)

        # Clean up extra spaces
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()

        # Get timing from first and last entries
        start_time = entries[0].start_time
        end_time = entries[-1].end_time

        # Get entry IDs
        entry_ids = [entry.id for entry in entries]

        return SentenceGroup(
            sentence_id=sentence_id,
            text=combined_text,
            start_time=start_time,
            end_time=end_time,
            subtitle_entries=entry_ids,
            duration_estimate=end_time - start_time,
            entry_count=len(entries)
        )

    def _save_debug_info(
        self,
        sentence_groups: List[SentenceGroup],
        debug_path: str
    ) -> None:
        """Save sentence grouping information for debugging."""
        debug_file = Path(debug_path) / "sentences.json"
        debug_file.parent.mkdir(parents=True, exist_ok=True)

        debug_data = {
            'total_sentences': len(sentence_groups),
            'sentences': [asdict(group) for group in sentence_groups],
            'statistics': self._calculate_statistics(sentence_groups)
        }

        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)

    def _calculate_statistics(self, sentence_groups: List[SentenceGroup]) -> dict:
        """Calculate statistics about the sentence groupings."""
        if not sentence_groups:
            return {}

        durations = [group.duration_estimate for group in sentence_groups]
        entry_counts = [group.entry_count for group in sentence_groups]
        text_lengths = [len(group.text) for group in sentence_groups]

        return {
            'total_sentences': len(sentence_groups),
            'average_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'average_entries_per_sentence': sum(entry_counts) / len(entry_counts),
            'average_text_length': sum(text_lengths) / len(text_lengths),
            'total_duration': max(group.end_time for group in sentence_groups),
        }

    def analyze_groupings(
        self,
        sentence_groups: List[SentenceGroup]
    ) -> dict:
        """Analyze sentence groupings and return insights."""
        if not sentence_groups:
            return {'error': 'No sentence groups provided'}

        # Basic statistics
        stats = self._calculate_statistics(sentence_groups)

        # Find sentences with unusual characteristics
        long_sentences = [
            group for group in sentence_groups
            if group.duration_estimate > 10.0
        ]

        short_sentences = [
            group for group in sentence_groups
            if group.duration_estimate < 1.0
        ]

        multi_entry_sentences = [
            group for group in sentence_groups
            if group.entry_count > 5
        ]

        return {
            'statistics': stats,
            'unusual_sentences': {
                'long_duration': len(long_sentences),
                'short_duration': len(short_sentences),
                'many_entries': len(multi_entry_sentences),
            },
            'examples': {
                'longest_sentence': max(sentence_groups, key=lambda x: x.duration_estimate).text[:100] + '...' if sentence_groups else '',
                'most_entries': max(sentence_groups, key=lambda x: x.entry_count).entry_count if sentence_groups else 0,
            }
        }


def main():
    """Test the sentence grouper with parsed SRT data."""
    from .srt_parser import SRTParser

    parser = SRTParser()
    grouper = SentenceGrouper()

    try:
        # Parse the SRT file
        entries = parser.parse_file('/mnt/d/Coloso/Syagamu/01.srt')
        print(f"Parsed {len(entries)} subtitle entries")

        # Group into sentences
        sentences = grouper.group_sentences(entries, save_debug_info=True, debug_path='./working')
        print(f"Grouped into {len(sentences)} sentences")

        # Show first few sentences
        for i, sentence in enumerate(sentences[:5]):
            print(f"\nSentence {sentence.sentence_id}:")
            print(f"  Time: {sentence.start_time:.3f} -> {sentence.end_time:.3f} ({sentence.duration_estimate:.2f}s)")
            print(f"  Entries: {sentence.entry_count} ({sentence.subtitle_entries})")
            print(f"  Text: {sentence.text[:100]}{'...' if len(sentence.text) > 100 else ''}")

        # Analyze groupings
        analysis = grouper.analyze_groupings(sentences)
        print(f"\nAnalysis:")
        print(f"  Average duration: {analysis['statistics']['average_duration']:.2f}s")
        print(f"  Average entries per sentence: {analysis['statistics']['average_entries_per_sentence']:.1f}")
        print(f"  Unusual sentences: {analysis['unusual_sentences']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()