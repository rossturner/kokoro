"""
Subtitle Extractor Module
Extracts embedded subtitle tracks from video files and converts them to SRT format.
"""

import subprocess
import json
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class SubtitleTrack:
    """Information about a subtitle track in a video file."""
    index: int
    codec: str
    language: Optional[str]
    title: Optional[str]
    disposition: Dict[str, Any]


class SubtitleExtractor:
    """Extracts subtitle tracks from video files using FFmpeg."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_subtitle_tracks(self, video_path: str) -> List[SubtitleTrack]:
        """
        Detect all subtitle tracks in a video file.

        Args:
            video_path: Path to the video file

        Returns:
            List of SubtitleTrack objects
        """
        try:
            # Use ffprobe to get stream information
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            subtitle_tracks = []
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'subtitle':
                    track = SubtitleTrack(
                        index=stream['index'],
                        codec=stream.get('codec_name', 'unknown'),
                        language=stream.get('tags', {}).get('language'),
                        title=stream.get('tags', {}).get('title'),
                        disposition=stream.get('disposition', {})
                    )
                    subtitle_tracks.append(track)

            return subtitle_tracks

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to detect subtitle tracks in {video_path}: {e}")
            return []

    def extract_subtitles(
        self,
        video_path: str,
        output_path: str,
        track_index: Optional[int] = None,
        language: Optional[str] = None
    ) -> bool:
        """
        Extract embedded subtitles to SRT format.

        Args:
            video_path: Path to the video file
            output_path: Path for the extracted SRT file
            track_index: Specific subtitle track index to extract
            language: Language code to extract (e.g., 'eng', 'spa')

        Returns:
            True if extraction was successful
        """
        # Detect available tracks
        tracks = self.detect_subtitle_tracks(video_path)
        if not tracks:
            self.logger.warning(f"No subtitle tracks found in {video_path}")
            return False

        # Select track to extract
        selected_track = self._select_track(tracks, track_index, language)
        if selected_track is None:
            self.logger.error("No suitable subtitle track found")
            return False

        try:
            # Extract subtitles using FFmpeg
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-map', f'0:s:{selected_track.index - self._get_first_subtitle_index(tracks)}',
                '-c:s', 'srt',
                str(output_path),
                '-y'  # Overwrite output file
            ]

            self.logger.info(f"Extracting subtitles from track {selected_track.index} ({selected_track.codec})")
            if selected_track.language:
                self.logger.info(f"Language: {selected_track.language}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Verify the output file was created and has content
            output_file = Path(output_path)
            if output_file.exists() and output_file.stat().st_size > 0:
                self.logger.info(f"Successfully extracted subtitles to {output_path}")

                # Clean extracted subtitles of HTML tags
                self._clean_extracted_subtitles(output_path)
                return True
            else:
                self.logger.error("Subtitle extraction failed - no output file created")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg failed to extract subtitles: {e}")
            if e.stderr:
                self.logger.error(f"FFmpeg error output: {e.stderr}")
            return False

    def _select_track(
        self,
        tracks: List[SubtitleTrack],
        track_index: Optional[int],
        language: Optional[str]
    ) -> Optional[SubtitleTrack]:
        """Select the best subtitle track based on criteria."""
        if track_index is not None:
            # Find track by absolute stream index
            for track in tracks:
                if track.index == track_index:
                    return track
            return None

        if language is not None:
            # Find track by language code
            for track in tracks:
                if track.language and track.language.lower() == language.lower():
                    return track
            return None

        # Default: return the first track, or the default track if marked
        default_track = None
        for track in tracks:
            if track.disposition.get('default', 0) == 1:
                default_track = track
                break

        return default_track if default_track else tracks[0]

    def _get_first_subtitle_index(self, tracks: List[SubtitleTrack]) -> int:
        """Get the absolute index of the first subtitle stream."""
        return min(track.index for track in tracks) if tracks else 0

    def _clean_extracted_subtitles(self, srt_path: str) -> None:
        """Clean HTML/XML tags from extracted subtitle file."""
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Clean HTML tags while preserving text content
            cleaned_content = self.clean_html_tags(content)

            # Only rewrite if content changed
            if cleaned_content != content:
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                self.logger.info("Cleaned HTML tags from extracted subtitles")

        except Exception as e:
            self.logger.warning(f"Failed to clean HTML tags from {srt_path}: {e}")

    def clean_html_tags(self, text: str) -> str:
        """
        Remove HTML/XML tags from text while preserving content.

        Args:
            text: Text potentially containing HTML tags

        Returns:
            Text with HTML tags removed
        """
        # Remove common subtitle HTML tags
        # Handle <font> tags with attributes
        text = re.sub(r'<font[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</font>', '', text, flags=re.IGNORECASE)

        # Handle other common tags
        text = re.sub(r'</?b>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?i>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?u>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?em>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?strong>', '', text, flags=re.IGNORECASE)

        # Handle tags with attributes (generic cleanup)
        text = re.sub(r'<[^>]+>', '', text)

        return text

    def get_subtitle_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get detailed information about subtitle tracks.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary with subtitle track information
        """
        tracks = self.detect_subtitle_tracks(video_path)

        info = {
            'has_subtitles': len(tracks) > 0,
            'track_count': len(tracks),
            'tracks': []
        }

        for track in tracks:
            track_info = {
                'index': track.index,
                'codec': track.codec,
                'language': track.language,
                'title': track.title,
                'default': track.disposition.get('default', 0) == 1
            }
            info['tracks'].append(track_info)

        return info


# Test functionality if run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python subtitle_extractor.py <video_path> [output.srt]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "extracted_subtitles.srt"

    extractor = SubtitleExtractor()

    # Show subtitle information
    info = extractor.get_subtitle_info(video_path)
    print(f"Video: {video_path}")
    print(f"Has subtitles: {info['has_subtitles']}")
    print(f"Track count: {info['track_count']}")

    for track in info['tracks']:
        print(f"  Track {track['index']}: {track['codec']}, "
              f"language={track['language']}, default={track['default']}")

    # Extract subtitles
    if info['has_subtitles']:
        success = extractor.extract_subtitles(video_path, output_path)
        if success:
            print(f"Successfully extracted subtitles to {output_path}")
        else:
            print("Failed to extract subtitles")
    else:
        print("No subtitles to extract")