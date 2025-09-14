"""
SRT Parser Module
Parses SRT subtitle files into structured Python objects.
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry from an SRT file."""
    id: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str          # cleaned text content
    raw_text: str      # original text with line breaks


class SRTParser:
    """Parser for SRT subtitle files."""

    def __init__(self):
        # Regex pattern for SRT timestamp format: 00:01:23,456 --> 00:01:25,789
        self.timestamp_pattern = re.compile(
            r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})'
        )

    def parse_file(self, srt_path: str, skip_subtitle_instructions: bool = True) -> List[SubtitleEntry]:
        """
        Parse an SRT file and return a list of SubtitleEntry objects.

        Args:
            srt_path: Path to the SRT file
            skip_subtitle_instructions: Skip entries containing subtitle instructions

        Returns:
            List of SubtitleEntry objects
        """
        srt_file = Path(srt_path)
        if not srt_file.exists():
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.parse_content(content, skip_subtitle_instructions)

    def parse_content(self, content: str, skip_subtitle_instructions: bool = True) -> List[SubtitleEntry]:
        """
        Parse SRT content string and return subtitle entries.

        Args:
            content: Raw SRT file content
            skip_subtitle_instructions: Skip entries containing subtitle instructions

        Returns:
            List of SubtitleEntry objects
        """
        entries = []
        blocks = self._split_into_blocks(content)

        for block in blocks:
            if not block.strip():
                continue

            entry = self._parse_block(block)
            if entry is None:
                continue

            # Skip subtitle instruction entries if requested
            if skip_subtitle_instructions and self._is_subtitle_instruction(entry.text):
                continue

            entries.append(entry)

        return entries

    def _split_into_blocks(self, content: str) -> List[str]:
        """Split SRT content into individual subtitle blocks."""
        # Split on double newlines, but handle different line ending types
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        blocks = re.split(r'\n\s*\n', content)
        return [block.strip() for block in blocks if block.strip()]

    def _parse_block(self, block: str) -> Optional[SubtitleEntry]:
        """Parse a single SRT block into a SubtitleEntry."""
        lines = block.split('\n')

        if len(lines) < 3:
            return None

        # First line should be the entry ID
        try:
            entry_id = int(lines[0].strip())
        except ValueError:
            return None

        # Second line should be the timestamp
        timestamp_match = self.timestamp_pattern.match(lines[1].strip())
        if not timestamp_match:
            return None

        start_time = self._timestamp_to_seconds(timestamp_match.groups()[:4])
        end_time = self._timestamp_to_seconds(timestamp_match.groups()[4:])

        # Remaining lines are the subtitle text
        raw_text = '\n'.join(lines[2:])
        cleaned_text = self._clean_text(raw_text)

        return SubtitleEntry(
            id=entry_id,
            start_time=start_time,
            end_time=end_time,
            text=cleaned_text,
            raw_text=raw_text
        )

    def _timestamp_to_seconds(self, timestamp_parts: tuple) -> float:
        """Convert timestamp parts (hours, minutes, seconds, milliseconds) to total seconds."""
        hours, minutes, seconds, milliseconds = map(int, timestamp_parts)
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

    def _clean_text(self, text: str) -> str:
        """Clean subtitle text by removing extra whitespace, HTML tags, and normalizing."""
        # Remove HTML/XML tags first
        text = self._remove_html_tags(text)

        # Replace newlines with spaces for multi-line entries
        text = re.sub(r'\n+', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML/XML tags from subtitle text while preserving content."""
        # Remove common subtitle HTML tags
        # Handle <font> tags with attributes
        text = re.sub(r'<font[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</font>', '', text, flags=re.IGNORECASE)

        # Handle other common formatting tags
        text = re.sub(r'</?b>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?i>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?u>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?em>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?strong>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?span[^>]*>', '', text, flags=re.IGNORECASE)

        # Generic cleanup for any remaining tags
        text = re.sub(r'<[^>]+>', '', text)

        return text

    def _is_subtitle_instruction(self, text: str) -> bool:
        """Check if the text appears to be a subtitle instruction rather than dialogue."""
        text_stripped = text.strip()
        # Only filter out entries that start with "If you want to turn off the subtitles"
        return text_stripped.startswith("If you want to turn off the subtitles")

    def get_total_duration(self, entries: List[SubtitleEntry]) -> float:
        """Get the total duration covered by subtitle entries."""
        if not entries:
            return 0.0

        return max(entry.end_time for entry in entries)

    def get_entry_count(self, entries: List[SubtitleEntry]) -> int:
        """Get the count of subtitle entries."""
        return len(entries)

    def validate_entries(self, entries: List[SubtitleEntry]) -> List[str]:
        """Validate subtitle entries and return list of any issues found."""
        issues = []

        for i, entry in enumerate(entries):
            # Check timing validity
            if entry.start_time >= entry.end_time:
                issues.append(f"Entry {entry.id}: Invalid timing (start >= end)")

            if entry.start_time < 0:
                issues.append(f"Entry {entry.id}: Negative start time")

            # Check for empty text
            if not entry.text.strip():
                issues.append(f"Entry {entry.id}: Empty text content")

            # Check for overlapping entries (optional warning)
            if i > 0 and entry.start_time < entries[i-1].end_time:
                issues.append(f"Entry {entry.id}: Overlaps with previous entry")

        return issues


def main():
    """Test the SRT parser with a sample file."""
    parser = SRTParser()

    # Test with the actual file
    try:
        entries = parser.parse_file('/mnt/d/Coloso/Syagamu/01.srt')
        print(f"Parsed {len(entries)} subtitle entries")
        print(f"Total duration: {parser.get_total_duration(entries):.2f} seconds")

        # Show first few entries
        for i, entry in enumerate(entries[:5]):
            print(f"\nEntry {entry.id}:")
            print(f"  Time: {entry.start_time:.3f} -> {entry.end_time:.3f}")
            print(f"  Duration: {entry.end_time - entry.start_time:.3f}s")
            print(f"  Text: {entry.text[:80]}{'...' if len(entry.text) > 80 else ''}")

        # Validate entries
        issues = parser.validate_entries(entries)
        if issues:
            print(f"\nValidation issues found: {len(issues)}")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"  - {issue}")
        else:
            print("\nNo validation issues found")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()