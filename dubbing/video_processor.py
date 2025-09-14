"""
Video Processor Module
Handles video file operations including audio extraction, replacement, and final video export.
"""

import os
import shutil
import subprocess
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .config import Config


@dataclass
class VideoInfo:
    """Information about a video file."""
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: Optional[str]
    bitrate: Optional[int]
    has_audio: bool


@dataclass
class ProcessingResult:
    """Result of video processing operations."""
    success: bool
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


class VideoProcessor:
    """Handles video file operations using FFmpeg."""

    def __init__(self, config: Config):
        self.config = config
        self._check_ffmpeg_available()
        self._gpu_available = self._check_gpu_available()

    def _check_ffmpeg_available(self) -> None:
        """Check if FFmpeg is available in the system."""
        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                   capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("FFmpeg not working properly")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg to use video processing features.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg check timed out")

    def _check_gpu_available(self) -> bool:
        """Check if GPU encoding (NVENC) is available."""
        try:
            # Check if the configured GPU codec is available
            result = subprocess.run([
                'ffmpeg', '-hide_banner', '-encoders'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                return False

            # Look for NVENC encoder in the output
            gpu_codec = self.config.gpu_codec
            if gpu_codec in result.stdout:
                if self.config.verbose_logging:
                    print(f"GPU encoder '{gpu_codec}' detected and available")
                return True
            else:
                if self.config.verbose_logging:
                    print(f"GPU encoder '{gpu_codec}' not available, falling back to CPU")
                return False

        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            if self.config.verbose_logging:
                print("Unable to detect GPU availability, using CPU encoding")
            return False

    @property
    def is_gpu_encoding_enabled(self) -> bool:
        """Check if GPU encoding is currently enabled and available."""
        return (self.config.enable_gpu_acceleration and
                self._gpu_available and
                self.config.enable_video_compression)

    def _get_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate MD5 hash of a file efficiently."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _files_are_identical(self, file1: Path, file2: Path) -> bool:
        """Check if two files are identical by comparing size and hash."""
        if not file1.exists() or not file2.exists():
            return False

        # Quick size check first
        if file1.stat().st_size != file2.stat().st_size:
            return False

        # For small files, compare hashes directly
        if file1.stat().st_size < 1024 * 1024:  # 1MB
            return self._get_file_hash(file1) == self._get_file_hash(file2)

        # For large files, compare modification time and size (heuristic)
        # This is much faster than hashing large video files
        stat1 = file1.stat()
        stat2 = file2.stat()

        # Same size and modification time suggests same file
        return (stat1.st_size == stat2.st_size and
                abs(stat1.st_mtime - stat2.st_mtime) < 1.0)

    def get_video_info(self, video_path: Path) -> VideoInfo:
        """
        Extract information about a video file using ffprobe.

        Args:
            video_path: Path to the video file

        Returns:
            VideoInfo object with video details
        """
        try:
            # Use ffprobe to get video information
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")

            import json
            probe_data = json.loads(result.stdout)

            # Extract video stream info
            video_stream = None
            audio_stream = None

            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'video' and video_stream is None:
                    video_stream = stream
                elif stream.get('codec_type') == 'audio' and audio_stream is None:
                    audio_stream = stream

            if not video_stream:
                raise RuntimeError("No video stream found")

            # Parse video information
            duration = float(probe_data.get('format', {}).get('duration', 0))
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))

            # Parse FPS (can be in different formats)
            fps_str = video_stream.get('r_frame_rate', '0/1')
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den) if float(den) != 0 else 0
            else:
                fps = float(fps_str)

            video_codec = video_stream.get('codec_name', 'unknown')

            # Audio information
            has_audio = audio_stream is not None
            audio_codec = audio_stream.get('codec_name') if audio_stream else None
            bitrate = int(probe_data.get('format', {}).get('bit_rate', 0)) if probe_data.get('format', {}).get('bit_rate') else None

            return VideoInfo(
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                video_codec=video_codec,
                audio_codec=audio_codec,
                bitrate=bitrate,
                has_audio=has_audio
            )

        except subprocess.TimeoutExpired:
            raise RuntimeError("Video info extraction timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to get video info: {e}")

    def copy_files_to_working_dir(
        self,
        video_path: str,
        srt_path: str,
        working_dir: Path
    ) -> Tuple[Path, Path, Dict[str, bool]]:
        """
        Copy video and SRT files to working directory, skipping if identical files exist.

        Args:
            video_path: Path to original video file
            srt_path: Path to original SRT file
            working_dir: Working directory path

        Returns:
            Tuple of (copied_video_path, copied_srt_path, copy_status_dict)
            copy_status_dict contains 'video_copied' and 'srt_copied' booleans
        """
        try:
            video_src = Path(video_path)
            srt_src = Path(srt_path)

            if not video_src.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            if not srt_src.exists():
                raise FileNotFoundError(f"SRT file not found: {srt_path}")

            # Create working directory if it doesn't exist
            working_dir.mkdir(parents=True, exist_ok=True)

            # Define destination paths
            video_dst = working_dir / video_src.name
            srt_dst = working_dir / "subtitles.srt"

            copy_status = {
                'video_copied': False,
                'srt_copied': False
            }

            # Check and copy video file
            if self._files_are_identical(video_src, video_dst):
                print(f"Video file already exists and is identical, skipping copy: {video_dst}")
            else:
                print(f"Copying video file: {video_src} -> {video_dst}")
                shutil.copy2(video_src, video_dst)
                copy_status['video_copied'] = True

            # Check and copy SRT file (always use hash for small text files)
            if srt_dst.exists() and self._get_file_hash(srt_src) == self._get_file_hash(srt_dst):
                print(f"SRT file already exists and is identical, skipping copy: {srt_dst}")
            else:
                print(f"Copying SRT file: {srt_src} -> {srt_dst}")
                shutil.copy2(srt_src, srt_dst)
                copy_status['srt_copied'] = True

            return video_dst, srt_dst, copy_status

        except Exception as e:
            raise RuntimeError(f"Failed to copy files to working directory: {e}")

    def strip_audio_from_video(
        self,
        input_video: Path,
        output_video: Path
    ) -> ProcessingResult:
        """
        Remove audio track from video file.

        Args:
            input_video: Input video file path
            output_video: Output video file path (without audio)

        Returns:
            ProcessingResult with operation details
        """
        try:
            import time
            start_time = time.time()

            cmd = [
                'ffmpeg',
                '-i', str(input_video),
                '-an',  # Remove audio
                '-c:v', 'copy',  # Copy video stream without re-encoding
                '-y',  # Overwrite output file
                str(output_video)
            ]

            if self.config.verbose_logging:
                print(f"Stripping audio: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            processing_time = time.time() - start_time

            if result.returncode != 0:
                return ProcessingResult(
                    success=False,
                    error_message=f"FFmpeg failed: {result.stderr}",
                    processing_time=processing_time
                )

            if not output_video.exists():
                return ProcessingResult(
                    success=False,
                    error_message="Output file was not created",
                    processing_time=processing_time
                )

            return ProcessingResult(
                success=True,
                output_path=output_video,
                processing_time=processing_time
            )

        except subprocess.TimeoutExpired:
            return ProcessingResult(
                success=False,
                error_message="Audio stripping timed out"
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Audio stripping failed: {e}"
            )

    def add_audio_to_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        audio_offset: float = 0.0
    ) -> ProcessingResult:
        """
        Add audio track to video file.

        Args:
            video_path: Video file path (without audio)
            audio_path: Audio file path to add
            output_path: Final output video path
            audio_offset: Audio offset in seconds (positive = delay audio)

        Returns:
            ProcessingResult with operation details
        """
        try:
            import time
            start_time = time.time()

            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',  # Copy video without re-encoding
                '-c:a', self.config.audio_codec,  # Audio codec
                '-shortest',  # End when shortest input ends
                '-y'  # Overwrite output file
            ]

            # Add audio offset if specified
            if audio_offset != 0.0:
                cmd.extend(['-itsoffset', str(audio_offset)])

            cmd.append(str(output_path))

            if self.config.verbose_logging:
                print(f"Adding audio: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            processing_time = time.time() - start_time

            if result.returncode != 0:
                return ProcessingResult(
                    success=False,
                    error_message=f"FFmpeg failed: {result.stderr}",
                    processing_time=processing_time
                )

            if not output_path.exists():
                return ProcessingResult(
                    success=False,
                    error_message="Output file was not created",
                    processing_time=processing_time
                )

            return ProcessingResult(
                success=True,
                output_path=output_path,
                processing_time=processing_time
            )

        except subprocess.TimeoutExpired:
            return ProcessingResult(
                success=False,
                error_message="Audio addition timed out"
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Audio addition failed: {e}"
            )

    def create_dubbed_video(
        self,
        original_video: Path,
        dubbed_audio: Path,
        output_path: Path,
        temp_dir: Optional[Path] = None
    ) -> ProcessingResult:
        """
        Create final dubbed video by replacing audio track.

        Args:
            original_video: Original video file
            dubbed_audio: Generated TTS audio file
            output_path: Final dubbed video output path
            temp_dir: Optional temporary directory for intermediate files

        Returns:
            ProcessingResult with operation details
        """
        try:
            # Use temp_dir or create one in the same directory as output
            if temp_dir is None:
                temp_dir = output_path.parent / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            temp_video = temp_dir / f"video_no_audio{original_video.suffix}"

            # Step 1: Strip original audio
            print("Step 1: Removing original audio...")
            strip_result = self.strip_audio_from_video(original_video, temp_video)

            if not strip_result.success:
                return ProcessingResult(
                    success=False,
                    error_message=f"Failed to strip audio: {strip_result.error_message}"
                )

            # Step 2: Add dubbed audio
            print("Step 2: Adding dubbed audio...")
            final_result = self.add_audio_to_video(temp_video, dubbed_audio, output_path)

            # Cleanup temporary file
            if temp_video.exists():
                temp_video.unlink()

            return final_result

        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Dubbed video creation failed: {e}"
            )

    def create_dubbed_video_with_compression_and_subtitles(
        self,
        original_video: Path,
        dubbed_audio: Path,
        srt_file: Path,
        output_path: Path,
        temp_dir: Optional[Path] = None
    ) -> ProcessingResult:
        """
        Create final dubbed video with compression and embedded subtitles.

        This method creates Handbrake-like output:
        - Replaces original audio with Kokoro TTS
        - Embeds SRT subtitles as mov_text stream
        - Applies video/audio compression for file size reduction
        - Maintains video quality with configurable settings

        Args:
            original_video: Original video file
            dubbed_audio: Generated TTS audio file
            srt_file: SRT subtitle file to embed
            output_path: Final output video path
            temp_dir: Optional temporary directory

        Returns:
            ProcessingResult with operation details
        """
        try:
            import time
            start_time = time.time()

            # Build FFmpeg command for compressed output with embedded subtitles
            cmd = self._build_compression_command(
                original_video, dubbed_audio, srt_file, output_path
            )

            if self.config.verbose_logging:
                print(f"Creating compressed video with subtitles: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for compression
            )

            processing_time = time.time() - start_time

            if result.returncode != 0:
                return ProcessingResult(
                    success=False,
                    error_message=f"FFmpeg compression failed: {result.stderr}",
                    processing_time=processing_time
                )

            if not output_path.exists():
                return ProcessingResult(
                    success=False,
                    error_message="Output file was not created",
                    processing_time=processing_time
                )

            return ProcessingResult(
                success=True,
                output_path=output_path,
                processing_time=processing_time
            )

        except subprocess.TimeoutExpired:
            return ProcessingResult(
                success=False,
                error_message="Video compression timed out"
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Compressed video creation failed: {e}"
            )

    def _build_compression_command(
        self,
        video_input: Path,
        audio_input: Path,
        subtitle_input: Path,
        output: Path
    ) -> list:
        """Build FFmpeg command for compressed output with embedded subtitles."""

        cmd = [
            'ffmpeg',
            '-i', str(video_input),        # Video input
            '-i', str(audio_input),        # Audio input (Kokoro TTS)
            '-i', str(subtitle_input),     # Subtitle input (SRT file)
        ]

        # Determine if we should use GPU or CPU encoding
        use_gpu = (self.config.enable_gpu_acceleration and
                  self._gpu_available and
                  self.config.enable_video_compression)

        if use_gpu:
            # GPU encoding settings (NVENC)
            cmd.extend([
                '-c:v', self.config.gpu_codec,
                '-preset', self.config.gpu_preset,
                '-profile:v', self.config.gpu_profile,
                '-rc', self.config.gpu_rc_mode,
                '-cq', str(self.config.gpu_cq)
            ])

            if self.config.verbose_logging:
                print(f"Using GPU encoding: {self.config.gpu_codec} with preset {self.config.gpu_preset}")
        else:
            # CPU encoding settings (traditional)
            cmd.extend([
                '-c:v', 'libx264',
                '-crf', str(self.config.video_crf),
                '-preset', self.config.video_preset
            ])

            if self.config.verbose_logging:
                print(f"Using CPU encoding: libx264 with preset {self.config.video_preset}")

        cmd.extend([
            # Audio compression settings
            '-c:a', self.config.audio_codec,  # AAC
            '-b:a', self.config.audio_compression_bitrate,

            # Subtitle embedding settings
            '-c:s', self.config.subtitle_codec,
            f'-metadata:s:s:0', f'language={self.config.subtitle_language}',

            # Stream mapping (video from input 0, audio from input 1, subtitles from input 2)
            '-map', '0:v',      # Original video
            '-map', '1:a',      # Kokoro TTS audio
            '-map', '2:s',      # SRT subtitles

            # Subtitle disposition
            '-disposition:s:0', 'default' if self.config.subtitle_default else '0',

            # Additional optimizations
            '-movflags', '+faststart',  # Enable fast start for web streaming
            '-shortest',  # End when shortest input ends

            '-y',  # Overwrite output
            str(output)
        ])

        # Add max bitrate only for CPU encoding (GPU handles this differently)
        if not use_gpu and hasattr(self.config, 'video_max_bitrate') and self.config.video_max_bitrate:
            # Insert max bitrate before the output filename
            output_index = cmd.index(str(output))
            cmd.insert(output_index, '-bufsize')
            cmd.insert(output_index + 1, '1000k')
            cmd.insert(output_index, '-maxrate')
            cmd.insert(output_index + 1, self.config.video_max_bitrate)

        return cmd

    def validate_video_audio_sync(
        self,
        video_path: Path,
        expected_duration: float,
        tolerance: float = 1.0
    ) -> Dict:
        """
        Validate that video and audio are properly synchronized.

        Args:
            video_path: Path to the video to validate
            expected_duration: Expected duration in seconds
            tolerance: Acceptable duration difference in seconds

        Returns:
            Dictionary with validation results
        """
        try:
            video_info = self.get_video_info(video_path)

            duration_diff = abs(video_info.duration - expected_duration)
            sync_ok = duration_diff <= tolerance

            return {
                'sync_ok': sync_ok,
                'video_duration': video_info.duration,
                'expected_duration': expected_duration,
                'duration_difference': duration_diff,
                'has_audio': video_info.has_audio,
                'audio_codec': video_info.audio_codec,
                'video_info': video_info
            }

        except Exception as e:
            return {
                'sync_ok': False,
                'error': str(e)
            }

    def cleanup_working_dir(self, working_dir: Path, keep_final_files: bool = True) -> None:
        """
        Clean up working directory, optionally keeping final output files.

        Args:
            working_dir: Working directory to clean
            keep_final_files: Whether to keep final output files
        """
        try:
            if not working_dir.exists():
                return

            files_to_keep = set()
            if keep_final_files:
                files_to_keep.update([
                    'final_audio.wav',
                    'output_video.mp4',
                    'timeline_debug.json',
                    'sentences.json'
                ])

            for item in working_dir.rglob('*'):
                if item.is_file():
                    if keep_final_files and item.name in files_to_keep:
                        continue
                    item.unlink()

            # Remove empty directories
            for item in working_dir.rglob('*'):
                if item.is_dir() and not any(item.iterdir()):
                    item.rmdir()

            if self.config.verbose_logging:
                print(f"Cleaned up working directory: {working_dir}")

        except Exception as e:
            print(f"Warning: Failed to cleanup working directory: {e}")

    def get_compression_stats(self, original_path: Path, compressed_path: Path) -> Dict:
        """Analyze compression effectiveness."""
        try:
            original_info = self.get_video_info(original_path)
            compressed_info = self.get_video_info(compressed_path)

            original_size = original_path.stat().st_size
            compressed_size = compressed_path.stat().st_size

            return {
                'original_size_mb': original_size / (1024 * 1024),
                'compressed_size_mb': compressed_size / (1024 * 1024),
                'size_reduction_percent': ((original_size - compressed_size) / original_size) * 100,
                'original_video_bitrate': original_info.bitrate,
                'compressed_video_bitrate': compressed_info.bitrate,
                'bitrate_reduction_percent': ((original_info.bitrate - compressed_info.bitrate) / original_info.bitrate) * 100 if original_info.bitrate else 0,
                'has_embedded_subtitles': self._has_subtitle_streams(compressed_path),
                'quality_maintained': compressed_info.width == original_info.width and compressed_info.height == original_info.height
            }
        except Exception as e:
            return {'error': str(e)}

    def validate_embedded_subtitles(self, video_path: Path) -> Dict:
        """Verify that subtitles are properly embedded."""
        try:
            # Use ffprobe to check for subtitle streams
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 's', str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return {'embedded': False, 'error': 'Failed to check subtitle streams'}

            import json
            probe_data = json.loads(result.stdout)

            subtitle_streams = probe_data.get('streams', [])

            if not subtitle_streams:
                return {'embedded': False, 'error': 'No subtitle streams found'}

            # Analyze first subtitle stream
            sub_stream = subtitle_streams[0]

            return {
                'embedded': True,
                'codec': sub_stream.get('codec_name'),
                'language': sub_stream.get('tags', {}).get('language'),
                'disposition': sub_stream.get('disposition', {}),
                'frame_count': int(sub_stream.get('nb_frames', 0))
            }

        except Exception as e:
            return {'embedded': False, 'error': str(e)}

    def _has_subtitle_streams(self, video_path: Path) -> bool:
        """Check if video has subtitle streams."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 's', str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return False

            import json
            probe_data = json.loads(result.stdout)

            return len(probe_data.get('streams', [])) > 0

        except Exception:
            return False


def main():
    """Test video processor functionality."""
    from .config import get_default_config

    config = get_default_config()
    config.verbose_logging = True

    processor = VideoProcessor(config)

    try:
        # Test with actual video file
        video_path = "/mnt/d/Coloso/Syagamu/01 Before Starting the Class.mp4"

        print("Getting video information...")
        video_info = processor.get_video_info(Path(video_path))

        print(f"Video Info:")
        print(f"  Duration: {video_info.duration:.2f}s")
        print(f"  Resolution: {video_info.width}x{video_info.height}")
        print(f"  FPS: {video_info.fps:.2f}")
        print(f"  Video Codec: {video_info.video_codec}")
        print(f"  Has Audio: {video_info.has_audio}")
        if video_info.has_audio:
            print(f"  Audio Codec: {video_info.audio_codec}")

        # Test file copying
        working_dir = Path("./working/test_video_processor")
        video_copy, srt_copy, copy_status = processor.copy_files_to_working_dir(
            video_path,
            "/mnt/d/Coloso/Syagamu/01.srt",
            working_dir
        )

        print(f"\nFiles copied to working directory:")
        print(f"  Video: {video_copy}")
        print(f"  SRT: {srt_copy}")

        # Test audio stripping
        video_no_audio = working_dir / "video_no_audio.mp4"
        print(f"\nStripping audio...")

        strip_result = processor.strip_audio_from_video(video_copy, video_no_audio)

        if strip_result.success:
            print(f"Audio stripped successfully in {strip_result.processing_time:.2f}s")
            print(f"Output: {strip_result.output_path}")

            # Verify the stripped video
            stripped_info = processor.get_video_info(video_no_audio)
            print(f"Stripped video has audio: {stripped_info.has_audio}")
        else:
            print(f"Audio stripping failed: {strip_result.error_message}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()