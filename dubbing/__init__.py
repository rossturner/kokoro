"""
Dubbing System Package
Video dubbing system using Kokoro TTS and intelligent sentence grouping.
"""

from .config import Config, get_default_config
from .srt_parser import SRTParser, SubtitleEntry
from .sentence_grouper import SentenceGrouper, SentenceGroup
from .tts_generator import TTSGenerator, AudioResult
from .audio_assembler import AudioAssembler, TimingEvent, AssemblyStats
from .video_processor import VideoProcessor, VideoInfo, ProcessingResult

__version__ = "1.0.0"
__all__ = [
    "Config",
    "get_default_config",
    "SRTParser",
    "SubtitleEntry",
    "SentenceGrouper",
    "SentenceGroup",
    "TTSGenerator",
    "AudioResult",
    "AudioAssembler",
    "TimingEvent",
    "AssemblyStats",
    "VideoProcessor",
    "VideoInfo",
    "ProcessingResult",
]