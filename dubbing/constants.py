"""
Constants Module
Central location for all hardcoded values used throughout the dubbing system.
"""

# Audio Processing Constants
AUDIO_FORMAT = 'wav'
AUDIO_SUBTYPE = 'PCM_16'
DEFAULT_SAMPLE_RATE = 24000
SUPPORTED_SAMPLE_RATES = [22050, 24000, 44100, 48000]

# Audio Normalization Constants
MAX_AMPLITUDE_THRESHOLD = 1.0
MIN_AMPLITUDE_THRESHOLD = 0.1
NORMALIZATION_TARGET = 0.95
QUIET_AUDIO_BOOST_TARGET = 0.7
SOFT_LIMITING_FACTOR = 0.9
SOFT_LIMITING_OUTPUT = 0.95

# Timing Constants
MIN_SENTENCE_GAP = 0.25  # 250ms minimum gap between sentences
MAX_SILENCE_DURATION = 5.0  # Maximum silence to insert (safety limit)
MIN_SILENCE_DURATION = 0.05  # Minimum silence gap (50ms)

# Quality Control Thresholds
MAX_AUDIO_DURATION_PER_SENTENCE = 30.0  # Maximum seconds for a single sentence
MIN_AUDIO_DURATION_PER_SENTENCE = 0.5   # Minimum seconds for a single sentence
LONG_SENTENCE_WARNING_THRESHOLD = 10.0   # Warn about sentences longer than this
HIGH_FAILURE_RATE_THRESHOLD = 0.1        # 10% failure rate threshold

# Video Processing Constants
DEFAULT_VIDEO_CODEC = 'libx264'
DEFAULT_AUDIO_CODEC = 'aac'
DEFAULT_VIDEO_CRF = 23
DEFAULT_VIDEO_PRESET = 'medium'
DEFAULT_AUDIO_BITRATE = '128k'
DEFAULT_VIDEO_MAX_BITRATE = '500k'

# Subtitle Constants
DEFAULT_SUBTITLE_CODEC = 'mov_text'
DEFAULT_SUBTITLE_LANGUAGE = 'eng'
SUPPORTED_SUBTITLE_CODECS = ['mov_text', 'srt', 'ass', 'webvtt', 'ttml']

# Compression Presets
COMPRESSION_PRESETS = {
    'archive': {
        'crf': 18,
        'preset': 'slow',
        'audio_br': '192k',
        'description': 'High quality archival'
    },
    'balanced': {
        'crf': 23,
        'preset': 'medium',
        'audio_br': '128k',
        'description': 'Handbrake-like balanced quality/size'
    },
    'compact': {
        'crf': 28,
        'preset': 'fast',
        'audio_br': '96k',
        'description': 'Smaller files for distribution'
    }
}

# TTS Constants
DEFAULT_VOICE = 'af_heart'
AVAILABLE_VOICES = [
    'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore',
    'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
    'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael',
    'am_onyx', 'am_puck', 'am_santa',
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
    'ef_dora', 'em_alex', 'em_santa', 'ff_siwis',
    'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi',
    'if_sara', 'im_nicola',
    'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo',
    'pf_dora', 'pm_alex', 'pm_santa',
    'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi',
    'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang'
]

# Text Cleaning Replacements for TTS
TTS_TEXT_REPLACEMENTS = {
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

# Sentence Detection
SENTENCE_ENDINGS = ['.', '!', '?']
COMMON_ABBREVIATIONS = [
    'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Lt.', 'Sgt.', 'Col.',
    'Ltd.', 'Inc.', 'Corp.', 'Co.', 'etc.', 'vs.', 'v.', 'vs',
    'i.e.', 'e.g.', 'a.m.', 'p.m.', 'A.M.', 'P.M.',
    'St.', 'Ave.', 'Blvd.', 'Rd.', 'Jr.', 'Sr.'
]

# Subtitle Instructions to Skip
SUBTITLE_INSTRUCTION_KEYWORDS = [
    'turn off the subtitles',
    'click [setting]',
    'tooltips',
    'setting',
    'close for mobile'
]

# File System Constants
DEFAULT_WORKING_DIR = 'working'
DEFAULT_OUTPUT_DIR = 'output'
AUDIO_SNIPPETS_DIR = 'audio_snippets'
DEBUG_DIR = 'debug'

# Progress Reporting
PROGRESS_REPORT_INTERVAL = 10  # Report progress every N items
PROGRESS_SUCCESS_RATE_DECIMALS = 1

# FFmpeg Timeout Values (in seconds)
FFPROBE_TIMEOUT = 30
FFMPEG_STRIP_AUDIO_TIMEOUT = 300  # 5 minutes
FFMPEG_ADD_AUDIO_TIMEOUT = 600    # 10 minutes
FFMPEG_COMPRESSION_TIMEOUT = 1800 # 30 minutes

# Performance Settings
MAX_CONCURRENT_TTS_JOBS = 1  # Keep at 1 for stable Kokoro processing
AUDIO_CHUNK_SIZE = 1024
RESAMPLING_CHUNK_SIZE = 4096

# Validation Tolerances
VIDEO_AUDIO_SYNC_TOLERANCE = 1.0  # seconds
DURATION_COMPARISON_TOLERANCE = 2.0  # seconds

# Logging Constants
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Magic Numbers to Named Constants
MIN_SENTENCE_WORD_COUNT = 3  # Minimum words for a complete sentence
VERY_QUIET_AUDIO_THRESHOLD = 0.1
CLIPPING_PREVENTION_FACTOR = 0.95
BUFFER_SIZE_KB = 1000  # For video max bitrate buffer