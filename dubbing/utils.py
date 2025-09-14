"""
Utilities Module
Shared utility functions used across the dubbing system.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from .constants import (
    MAX_AMPLITUDE_THRESHOLD,
    MIN_AMPLITUDE_THRESHOLD,
    NORMALIZATION_TARGET,
    QUIET_AUDIO_BOOST_TARGET,
    SOFT_LIMITING_FACTOR,
    SOFT_LIMITING_OUTPUT,
    VERY_QUIET_AUDIO_THRESHOLD,
    CLIPPING_PREVENTION_FACTOR
)

logger = logging.getLogger(__name__)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to prevent clipping and ensure proper range.

    Args:
        audio: Input audio array

    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio

    # Find peak amplitude
    max_amplitude = max(abs(audio.max()), abs(audio.min()))

    if max_amplitude > MAX_AMPLITUDE_THRESHOLD:
        # Normalize to prevent clipping
        audio = audio * (CLIPPING_PREVENTION_FACTOR / max_amplitude)
        logger.debug(f"Normalized audio to prevent clipping (peak: {max_amplitude:.3f})")
    elif max_amplitude < MIN_AMPLITUDE_THRESHOLD:
        # Boost very quiet audio (but be conservative)
        audio = audio * (QUIET_AUDIO_BOOST_TARGET / max_amplitude)
        logger.debug(f"Boosted quiet audio (peak: {max_amplitude:.3f})")

    # Apply soft limiting to prevent any remaining issues
    audio = np.tanh(audio * SOFT_LIMITING_FACTOR) * SOFT_LIMITING_OUTPUT

    return audio


def normalize_final_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize the final assembled audio to optimal levels.

    Args:
        audio: Final assembled audio array

    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio

    # Find peak amplitude
    max_amplitude = max(abs(audio.max()), abs(audio.min()))

    if max_amplitude > NORMALIZATION_TARGET:
        # Normalize to prevent clipping
        audio = audio * (NORMALIZATION_TARGET / max_amplitude)
        logger.info(f"Normalized final audio to prevent clipping (peak: {max_amplitude:.3f})")
    elif max_amplitude < VERY_QUIET_AUDIO_THRESHOLD:
        # Boost very quiet audio
        audio = audio * (QUIET_AUDIO_BOOST_TARGET / max_amplitude)
        logger.info(f"Boosted quiet final audio (peak: {max_amplitude:.3f})")

    return audio


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """
    Simple audio resampling using linear interpolation.

    Args:
        audio: Input audio array
        from_rate: Source sample rate
        to_rate: Target sample rate

    Returns:
        Resampled audio array
    """
    if from_rate == to_rate:
        return audio

    # Calculate resampling ratio
    ratio = to_rate / from_rate
    new_length = int(len(audio) * ratio)

    # Simple linear interpolation resampling
    old_indices = np.linspace(0, len(audio) - 1, new_length)
    resampled = np.interp(old_indices, np.arange(len(audio)), audio)

    logger.debug(f"Resampled audio from {from_rate}Hz to {to_rate}Hz")
    return resampled.astype(np.float32)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for filesystem safety.

    Args:
        filename: Input filename

    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    safe_chars = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.'))
    safe_chars = safe_chars.replace(' ', '_').strip()

    # Ensure it's not empty
    if not safe_chars:
        safe_chars = "untitled"

    return safe_chars


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Set up logging configuration for the dubbing system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    from .constants import LOG_FORMAT, LOG_DATE_FORMAT

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[]
    )

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

    # Add file handler if specified
    handlers = [console_handler]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        handlers.append(file_handler)

    # Configure dubbing logger
    dubbing_logger = logging.getLogger('dubbing')
    dubbing_logger.handlers = handlers
    dubbing_logger.propagate = False


def validate_file_exists(file_path: Path, file_type: str = "file") -> None:
    """
    Validate that a file exists and raise descriptive error if not.

    Args:
        file_path: Path to validate
        file_type: Description of file type for error message

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{file_type.title()} not found: {file_path}")


def validate_sample_rate(sample_rate: int) -> bool:
    """
    Validate that a sample rate is supported.

    Args:
        sample_rate: Sample rate to validate

    Returns:
        True if valid
    """
    from .constants import SUPPORTED_SAMPLE_RATES
    return sample_rate in SUPPORTED_SAMPLE_RATES


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "1m 23s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def calculate_success_rate(successful: int, total: int) -> float:
    """
    Calculate success rate percentage.

    Args:
        successful: Number of successful operations
        total: Total number of operations

    Returns:
        Success rate as percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (successful / total) * 100


def create_progress_callback(logger_name: str = "dubbing.progress"):
    """
    Create a progress callback function for long-running operations.

    Args:
        logger_name: Logger name to use

    Returns:
        Progress callback function
    """
    progress_logger = logging.getLogger(logger_name)

    def progress_callback(current: int, total: int, result: Any = None) -> None:
        from .constants import PROGRESS_REPORT_INTERVAL, PROGRESS_SUCCESS_RATE_DECIMALS

        if current % PROGRESS_REPORT_INTERVAL == 0 or current == total:
            percent = (current / total) * 100
            status = "SUCCESS" if (result and hasattr(result, 'success') and result.success) else "PROCESSED"
            progress_logger.info(f"Progress: {current}/{total} ({percent:.{PROGRESS_SUCCESS_RATE_DECIMALS}f}%) - Last: {status}")

    return progress_callback


def ensure_directory(directory: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path to create
    """
    directory.mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / (1024 * 1024)


class ProgressReporter:
    """Context manager for reporting progress on long operations."""

    def __init__(self, operation_name: str, total_items: int, logger_name: str = "dubbing"):
        self.operation_name = operation_name
        self.total_items = total_items
        self.current_item = 0
        self.logger = logging.getLogger(logger_name)

    def __enter__(self):
        self.logger.info(f"Starting {self.operation_name} ({self.total_items} items)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name}")
        else:
            self.logger.error(f"Failed {self.operation_name}: {exc_val}")

    def update(self, increment: int = 1) -> None:
        """Update progress by increment."""
        self.current_item += increment
        if self.current_item % 10 == 0 or self.current_item == self.total_items:
            percent = (self.current_item / self.total_items) * 100
            self.logger.info(f"{self.operation_name}: {self.current_item}/{self.total_items} ({percent:.1f}%)")


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for optimal TTS generation.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text suitable for TTS
    """
    from .constants import TTS_TEXT_REPLACEMENTS

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Apply replacements
    for old, new in TTS_TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Ensure proper sentence ending
    if text and not text.endswith(('.', '!', '?')):
        text += '.'

    return text