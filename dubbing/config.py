"""
Configuration Module
Central configuration settings for the dubbing system.
"""

from pathlib import Path
from .constants import DEFAULT_VOICE, AVAILABLE_VOICES, DEFAULT_SAMPLE_RATE

# TTS Configuration
SAMPLE_RATE = DEFAULT_SAMPLE_RATE

# Timing Configuration
MIN_SENTENCE_GAP = 0.250           # 250ms minimum gap between sentences when overflow occurs
MAX_SILENCE_DURATION = 5.0        # Maximum silence to insert (safety limit)
MIN_SILENCE_DURATION = 0.05       # Minimum silence gap (50ms)

# Audio Processing Configuration
AUDIO_FORMAT = 'wav'
AUDIO_BITRATE = '192k'

# Video Processing Configuration
VIDEO_CODEC = 'libx264'
AUDIO_CODEC = 'aac'

# GPU Acceleration Settings
ENABLE_GPU_ACCELERATION = True       # Use GPU encoding when available
GPU_CODEC = 'h264_nvenc'            # NVIDIA NVENC encoder (h264_nvenc, hevc_nvenc)
GPU_PRESET = 'p4'                   # NVENC preset: p1-p7 (p1=fastest, p7=slowest/best quality)
GPU_PROFILE = 'high'                # NVENC profile: baseline, main, high
GPU_RC_MODE = 'vbr'                 # Rate control: cbr, vbr, constqp, lossless
GPU_CQ = 23                         # Constant quality (0-51, lower=better)

# Video Compression Settings (Default: Enabled for Handbrake-equivalent output)
ENABLE_VIDEO_COMPRESSION = True
VIDEO_CRF = 23                      # Constant Rate Factor (0-51, lower=better quality)
VIDEO_PRESET = 'medium'             # Encoding speed: ultrafast, fast, medium, slow, veryslow
VIDEO_MAX_BITRATE = '500k'          # Maximum video bitrate for consistent file sizes
AUDIO_COMPRESSION_BITRATE = '128k'  # Audio compression (matches Handbrake output)

# Subtitle Embedding Settings (Default: Enabled for embedded English subtitles)
EMBED_SUBTITLES = True
SUBTITLE_CODEC = 'mov_text'         # mov_text (MP4), srt, ass, webvtt, ttml
SUBTITLE_LANGUAGE = 'eng'           # ISO 639-2 language code
SUBTITLE_DEFAULT = True             # Make subtitle stream default
SUBTITLE_FORCED = False             # Whether subtitles are forced

# Quality Presets
COMPRESSION_PRESETS = {
    'archive': {'crf': 18, 'preset': 'slow', 'audio_br': '192k'},      # High quality
    'balanced': {'crf': 23, 'preset': 'medium', 'audio_br': '128k'},   # Default (Handbrake-like)
    'compact': {'crf': 28, 'preset': 'fast', 'audio_br': '96k'},       # Smaller files
}

# GPU Quality Presets (NVENC)
GPU_COMPRESSION_PRESETS = {
    'gpu_archive': {'cq': 18, 'preset': 'p6', 'audio_br': '192k'},     # High quality GPU
    'gpu_balanced': {'cq': 23, 'preset': 'p4', 'audio_br': '128k'},    # Default GPU (fast)
    'gpu_fast': {'cq': 26, 'preset': 'p2', 'audio_br': '128k'},        # Very fast GPU
}

# Processing Modes
SENTENCE_MODE = True               # Use sentence grouping vs entry-by-entry
SKIP_SUBTITLE_INSTRUCTIONS = True # Skip "turn off subtitles" entries

# Directory Configuration
DEFAULT_WORKING_DIR = './working'
DEFAULT_OUTPUT_DIR = './output'
CLEANUP_INTERMEDIATE_FILES = False # Keep files for debugging

# Sentence Detection Configuration
SENTENCE_ENDINGS = ['.', '!', '?']
ABBREVIATIONS = [
    'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Lt.', 'Sgt.', 'Col.',
    'Ltd.', 'Inc.', 'Corp.', 'Co.', 'etc.', 'vs.', 'v.', 'vs',
    'i.e.', 'e.g.', 'a.m.', 'p.m.', 'A.M.', 'P.M.',
    'St.', 'Ave.', 'Blvd.', 'Rd.', 'Jr.', 'Sr.'
]

# Quality Control Settings
MAX_AUDIO_DURATION_PER_SENTENCE = 30.0  # Maximum seconds for a single sentence
MIN_AUDIO_DURATION_PER_SENTENCE = 0.5   # Minimum seconds for a single sentence

# Logging Configuration
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
VERBOSE_LOGGING = False

# Performance Settings
MAX_CONCURRENT_TTS_JOBS = 1       # Keep at 1 for stable Kokoro processing
AUDIO_CHUNK_SIZE = 1024           # Buffer size for audio processing


class Config:
    """Configuration class with validation and helper methods."""

    def __init__(self):
        self.voice = DEFAULT_VOICE
        self.sample_rate = SAMPLE_RATE
        self.min_sentence_gap = MIN_SENTENCE_GAP
        self.max_silence_duration = MAX_SILENCE_DURATION
        self.sentence_mode = SENTENCE_MODE
        self.skip_subtitle_instructions = SKIP_SUBTITLE_INSTRUCTIONS
        self.working_dir = Path(DEFAULT_WORKING_DIR)
        self.output_dir = Path(DEFAULT_OUTPUT_DIR)
        self.cleanup_intermediate_files = CLEANUP_INTERMEDIATE_FILES
        self.verbose_logging = VERBOSE_LOGGING

        # GPU acceleration settings
        self.enable_gpu_acceleration = ENABLE_GPU_ACCELERATION
        self.gpu_codec = GPU_CODEC
        self.gpu_preset = GPU_PRESET
        self.gpu_profile = GPU_PROFILE
        self.gpu_rc_mode = GPU_RC_MODE
        self.gpu_cq = GPU_CQ

        # New compression settings
        self.enable_video_compression = ENABLE_VIDEO_COMPRESSION
        self.video_crf = VIDEO_CRF
        self.video_preset = VIDEO_PRESET
        self.video_max_bitrate = VIDEO_MAX_BITRATE
        self.audio_compression_bitrate = AUDIO_COMPRESSION_BITRATE

        # Video/Audio codecs
        self.video_codec = VIDEO_CODEC
        self.audio_codec = AUDIO_CODEC

        # New subtitle settings
        self.embed_subtitles = EMBED_SUBTITLES
        self.subtitle_codec = SUBTITLE_CODEC
        self.subtitle_language = SUBTITLE_LANGUAGE
        self.subtitle_default = SUBTITLE_DEFAULT
        self.subtitle_forced = SUBTITLE_FORCED

        # Compression preset
        self.compression_preset = 'balanced'

    def validate(self) -> list:
        """Validate configuration settings and return any issues."""
        issues = []

        # Validate voice
        if self.voice not in AVAILABLE_VOICES:
            issues.append(f"Invalid voice '{self.voice}'. Available: {AVAILABLE_VOICES}")

        # Validate sample rate
        if self.sample_rate not in [22050, 24000, 44100, 48000]:
            issues.append(f"Unusual sample rate: {self.sample_rate}Hz")

        # Validate timing settings
        if self.min_sentence_gap < 0:
            issues.append("min_sentence_gap cannot be negative")

        if self.max_silence_duration < self.min_sentence_gap:
            issues.append("max_silence_duration must be >= min_sentence_gap")

        # Validate compression settings
        if self.video_crf < 0 or self.video_crf > 51:
            issues.append("video_crf must be between 0-51")

        if self.video_preset not in ['ultrafast', 'fast', 'medium', 'slow', 'veryslow']:
            issues.append(f"Invalid video_preset '{self.video_preset}'")

        if self.subtitle_codec not in ['mov_text', 'srt', 'ass', 'webvtt', 'ttml']:
            issues.append(f"Invalid subtitle_codec '{self.subtitle_codec}'")

        # Validate GPU settings
        if self.gpu_codec not in ['h264_nvenc', 'hevc_nvenc']:
            issues.append(f"Invalid gpu_codec '{self.gpu_codec}'")

        if self.gpu_preset not in ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']:
            issues.append(f"Invalid gpu_preset '{self.gpu_preset}'")

        if self.gpu_profile not in ['baseline', 'main', 'high']:
            issues.append(f"Invalid gpu_profile '{self.gpu_profile}'")

        if self.gpu_rc_mode not in ['cbr', 'vbr', 'constqp', 'lossless']:
            issues.append(f"Invalid gpu_rc_mode '{self.gpu_rc_mode}'")

        if self.gpu_cq < 0 or self.gpu_cq > 51:
            issues.append("gpu_cq must be between 0-51")

        return issues

    def create_working_dir(self, video_name: str) -> Path:
        """Create and return working directory for a specific video."""
        # Sanitize video name for filesystem
        safe_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')

        working_path = self.working_dir / safe_name
        working_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (working_path / 'audio_snippets').mkdir(exist_ok=True)

        return working_path

    def get_audio_snippet_path(self, working_dir: Path, sentence_id: int) -> Path:
        """Get path for an individual audio snippet."""
        return working_dir / 'audio_snippets' / f'sentence_{sentence_id:03d}.wav'

    def get_final_audio_path(self, working_dir: Path) -> Path:
        """Get path for the final combined audio file."""
        return working_dir / 'final_audio.wav'

    def get_debug_info_path(self, working_dir: Path) -> Path:
        """Get path for debug information files."""
        return working_dir / 'debug'

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'voice': self.voice,
            'sample_rate': self.sample_rate,
            'min_sentence_gap': self.min_sentence_gap,
            'max_silence_duration': self.max_silence_duration,
            'sentence_mode': self.sentence_mode,
            'skip_subtitle_instructions': self.skip_subtitle_instructions,
            'working_dir': str(self.working_dir),
            'cleanup_intermediate_files': self.cleanup_intermediate_files,
            'verbose_logging': self.verbose_logging,
            'enable_video_compression': self.enable_video_compression,
            'video_crf': self.video_crf,
            'video_preset': self.video_preset,
            'video_max_bitrate': self.video_max_bitrate,
            'audio_compression_bitrate': self.audio_compression_bitrate,
            'video_codec': self.video_codec,
            'audio_codec': self.audio_codec,
            'embed_subtitles': self.embed_subtitles,
            'subtitle_codec': self.subtitle_codec,
            'subtitle_language': self.subtitle_language,
            'subtitle_default': self.subtitle_default,
            'subtitle_forced': self.subtitle_forced,
            'compression_preset': self.compression_preset
        }

    def from_dict(self, config_dict: dict) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if key == 'working_dir':
                    self.working_dir = Path(value)
                elif key == 'output_dir':
                    self.output_dir = Path(value)
                else:
                    setattr(self, key, value)

    def apply_compression_preset(self, preset_name: str) -> None:
        """Apply a compression preset."""
        if preset_name in COMPRESSION_PRESETS:
            preset = COMPRESSION_PRESETS[preset_name]
            self.video_crf = preset['crf']
            self.video_preset = preset['preset']
            self.audio_compression_bitrate = preset['audio_br']
            self.compression_preset = preset_name

    def apply_gpu_preset(self, preset_name: str) -> None:
        """Apply a GPU compression preset."""
        if preset_name in GPU_COMPRESSION_PRESETS:
            preset = GPU_COMPRESSION_PRESETS[preset_name]
            self.gpu_cq = preset['cq']
            self.gpu_preset = preset['preset']
            self.audio_compression_bitrate = preset['audio_br']
            self.compression_preset = preset_name

    def __str__(self) -> str:
        """String representation of configuration."""
        config_lines = [
            f"Voice: {self.voice}",
            f"Sample Rate: {self.sample_rate}Hz",
            f"Min Sentence Gap: {self.min_sentence_gap}s",
            f"Sentence Mode: {self.sentence_mode}",
            f"Working Directory: {self.working_dir}",
            f"Cleanup Files: {self.cleanup_intermediate_files}",
            f"Video Compression: {self.enable_video_compression}",
            f"GPU Acceleration: {self.enable_gpu_acceleration}",
            f"GPU Codec: {self.gpu_codec}",
            f"GPU Quality (CQ): {self.gpu_cq}",
            f"GPU Preset: {self.gpu_preset}",
            f"Video Quality (CRF): {self.video_crf}",
            f"Video Preset: {self.video_preset}",
            f"Audio Bitrate: {self.audio_compression_bitrate}",
            f"Embed Subtitles: {self.embed_subtitles}",
            f"Subtitle Codec: {self.subtitle_codec}",
        ]
        return "Configuration:\n  " + "\n  ".join(config_lines)


def get_default_config() -> Config:
    """Get default configuration instance."""
    return Config()


def load_config_from_file(config_path: str) -> Config:
    """Load configuration from a JSON file."""
    import json

    config = Config()
    config_file = Path(config_path)

    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config.from_dict(config_dict)

    return config


def save_config_to_file(config: Config, config_path: str) -> None:
    """Save configuration to a JSON file."""
    import json

    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2)


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print(config)

    # Test validation
    issues = config.validate()
    if issues:
        print("\nConfiguration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration is valid")

    # Test working directory creation
    test_dir = config.create_working_dir("Test Video Name")
    print(f"\nCreated working directory: {test_dir}")

    # Show paths
    print(f"Audio snippet path: {config.get_audio_snippet_path(test_dir, 1)}")
    print(f"Final audio path: {config.get_final_audio_path(test_dir)}")
    print(f"Debug info path: {config.get_debug_info_path(test_dir)}")