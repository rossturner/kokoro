# Video Dubbing System with Kokoro TTS - Comprehensive Plan

## Overview

This system creates dubbed versions of video files using Kokoro TTS and subtitle (.srt) files. The key innovation is intelligent sentence grouping - instead of generating audio for individual subtitle entries, the system groups consecutive entries that form complete sentences, resulting in more natural-sounding speech.

## System Architecture

### Directory Structure

```
/mnt/d/workspace/kokoro/
├── dubbing/                    # Dubbing system source code
│   ├── srt_parser.py           # Parse SRT files into structured data
│   ├── sentence_grouper.py     # Group subtitle entries into sentences
│   ├── tts_generator.py        # Generate TTS audio using Kokoro
│   ├── audio_assembler.py      # Assemble audio with intelligent timing
│   ├── video_processor.py      # Handle video file operations
│   ├── main.py                 # Main orchestrator script
│   └── config.py               # Configuration settings
├── working/                    # Working directory for processing
│   └── [video_name]/           # Per-video working directory
│       ├── video.mp4           # Copied original video file
│       ├── subtitles.srt       # Copied original SRT file
│       ├── sentences.json      # Parsed sentence groups (debug/info)
│       ├── audio_snippets/     # Individual TTS audio files
│       │   ├── sentence_001.wav
│       │   ├── sentence_002.wav
│       │   └── ...
│       ├── final_audio.wav     # Combined audio track
│       └── output_video.mp4    # Final dubbed video
└── DUBBING_SYSTEM.md           # This documentation file
```

## Core Components

### 1. SRT Parser (`srt_parser.py`)

**Purpose**: Parse SRT subtitle files into structured Python objects.

**Key Features**:
- Parse standard SRT format with entry numbers, timestamps, and text
- Skip first entry if it contains "turn off the subtitles" instruction
- Convert timestamps to floating-point seconds for easier processing
- Clean and normalize text content
- Handle multi-line subtitle entries

**Data Structure**:
```python
SubtitleEntry:
    - id: int (entry number)
    - start_time: float (seconds)
    - end_time: float (seconds)
    - text: str (cleaned text content)
    - raw_text: str (original text with line breaks)
```

### 2. Sentence Grouper (`sentence_grouper.py`)

**Purpose**: Intelligently group consecutive subtitle entries that form complete sentences.

**Algorithm**:
1. Iterate through subtitle entries in order
2. Accumulate entries until detecting sentence completion
3. Sentence completion detected by:
   - Ending punctuation (. ! ?)
   - Excluding common abbreviations (Dr., Mr., Ms., Mrs., etc., vs.)
   - Considering capitalization patterns of next entry
4. Create sentence groups with combined text and timing spans

**Sentence Detection Logic**:
```python
def is_sentence_end(text):
    text = text.strip()
    if text.endswith(('.', '!', '?')):
        # Exclude common abbreviations
        abbreviations = ['Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Ltd.', 'Inc.', 'etc.', 'vs.', 'i.e.', 'e.g.']
        if not any(text.endswith(abbr) for abbr in abbreviations):
            return True
    return False
```

**Output Structure**:
```python
SentenceGroup:
    - sentence_id: int
    - text: str (complete sentence)
    - start_time: float (from first subtitle entry)
    - end_time: float (from last subtitle entry)
    - subtitle_entries: List[int] (original entry IDs)
    - duration_estimate: float (end_time - start_time)
```

**Example Processing**:
```
Input Subtitles:
8. (30.505-34.189) "I work as a freelance illustrator mainly doing"
9. (34.189-37.874) "character-centered illustrations and character"
10. (37.874-42.441) "designs for casual games and virtual YouTuber industries."

Output Sentence:
{
    "sentence_id": 1,
    "text": "I work as a freelance illustrator mainly doing character-centered illustrations and character designs for casual games and virtual YouTuber industries.",
    "start_time": 30.505,
    "end_time": 42.441,
    "subtitle_entries": [8, 9, 10],
    "duration_estimate": 11.936
}
```

### 3. TTS Generator (`tts_generator.py`)

**Purpose**: Generate audio snippets for complete sentences using Kokoro TTS.

**Key Features**:
- Initialize Kokoro pipeline with configurable voice (default: `af_heart`)
- Generate audio for each sentence group
- Save individual audio files for debugging/reuse
- Return actual audio duration for timing calculations
- Handle TTS errors gracefully

**Process**:
1. Initialize Kokoro pipeline once at startup
2. For each sentence group:
   - Clean text (remove extra whitespace, normalize punctuation)
   - Generate TTS audio using Kokoro
   - Save as WAV file at 24kHz
   - Measure actual audio duration
   - Return audio data and duration

**Audio File Naming**:
- `sentence_001.wav`, `sentence_002.wav`, etc.
- Sequential numbering based on sentence group order

### 4. Audio Assembler (`audio_assembler.py`)

**Purpose**: Assemble individual sentence audio snippets into a complete audio track with intelligent timing.

**Core Algorithm - Timeline-Based Assembly**:

```python
current_time = 0.0
min_gap = 0.250  # 250ms minimum gap between sentences when overflowing

for sentence in sentences:
    ideal_start = sentence.start_time

    if current_time <= ideal_start:
        # We can start on time - insert silence to catch up if needed
        if current_time < ideal_start:
            insert_silence(duration=ideal_start - current_time)
        actual_start = ideal_start
    else:
        # Previous audio is still playing - add minimum gap
        actual_start = current_time + min_gap

    # Place the sentence audio
    audio_duration = place_audio(sentence.audio, at=actual_start)
    current_time = actual_start + audio_duration

    # Log timing for debugging
    log_timing(sentence, ideal_start, actual_start, audio_duration)
```

**Timing Scenarios**:

1. **On Time**: Audio finishes before next sentence starts
   - Insert silence to maintain timing
   - Start next sentence at its designated time

2. **Slight Overflow**: Audio extends slightly past next sentence start
   - Add 250ms gap after current audio ends
   - Start next sentence after the gap

3. **Significant Overflow**: Audio runs well past next sentence
   - Audio continues naturally
   - Next sentence starts after gap
   - Natural subtitle gaps allow "catch-up" later

**Output**:
- `final_audio.wav` at 24kHz sample rate
- Timing log for debugging and analysis

### 5. Video Processor (`video_processor.py`)

**Purpose**: Handle video file operations including audio replacement.

**Operations**:

1. **File Management**:
   - Create working directory for each video
   - Copy original video and SRT files (no modification of originals)
   - Clean up intermediate files (optional)

2. **Audio Extraction/Replacement**:
   - Use ffmpeg to strip original audio track
   - Replace with generated TTS audio track
   - Maintain original video quality and settings
   - Handle different video formats and codecs

3. **Final Video Export**:
   - Merge new audio with original video
   - Export as MP4 with appropriate codecs
   - Preserve video metadata where possible

**FFmpeg Commands**:
```bash
# Strip original audio
ffmpeg -i input_video.mp4 -an -c:v copy video_no_audio.mp4

# Add new audio track
ffmpeg -i video_no_audio.mp4 -i final_audio.wav \
       -c:v copy -c:a aac -shortest output_video.mp4
```

### 6. Main Orchestrator (`main.py`)

**Purpose**: Coordinate all components and provide user interface.

**Workflow**:

1. **Setup Phase**:
   - Parse command line arguments
   - Validate input files exist
   - Create working directory structure
   - Copy source files

2. **Parsing Phase**:
   - Parse SRT file into subtitle entries
   - Group entries into sentences
   - Save sentence groups to JSON for debugging

3. **TTS Generation Phase**:
   - Initialize Kokoro pipeline
   - Generate audio for each sentence
   - Save individual audio files
   - Track progress and errors

4. **Audio Assembly Phase**:
   - Assemble sentences into complete audio track
   - Apply timing algorithm with overflow handling
   - Generate final audio file

5. **Video Processing Phase**:
   - Strip original audio from video
   - Merge new audio track
   - Export final dubbed video

6. **Cleanup Phase**:
   - Optionally clean intermediate files
   - Report timing statistics and any issues

**Command Line Interface**:
```bash
python dubbing/main.py \
  --video "/path/to/video.mp4" \
  --srt "/path/to/subtitles.srt" \
  --working-dir "./working/video_name" \
  [--voice af_heart] \
  [--min-gap 0.25] \
  [--mode sentence] \
  [--cleanup false]
```

### 7. Configuration (`config.py`)

**Settings Categories**:

```python
# TTS Configuration
DEFAULT_VOICE = 'af_heart'
SAMPLE_RATE = 24000
AVAILABLE_VOICES = ['af_heart', 'af_sky', 'af_bella', 'am_adam', 'am_michael']

# Timing Configuration
MIN_SENTENCE_GAP = 0.250      # 250ms gap when overflow occurs
MAX_SILENCE_DURATION = 5.0    # Maximum silence to insert (safety)
SENTENCE_MODE = True          # Use sentence grouping vs entry-by-entry

# Processing Configuration
SKIP_SUBTITLE_INSTRUCTIONS = True  # Skip "turn off subtitles" entries
WORKING_DIR = './working'
CLEANUP_INTERMEDIATE_FILES = False  # Keep files for debugging

# Audio Configuration
AUDIO_FORMAT = 'wav'
AUDIO_BITRATE = '192k'
VIDEO_CODEC = 'libx264'
AUDIO_CODEC = 'aac'

# Sentence Detection
SENTENCE_ENDINGS = ['.', '!', '?']
ABBREVIATIONS = ['Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Ltd.', 'Inc.',
                'etc.', 'vs.', 'i.e.', 'e.g.', 'a.m.', 'p.m.']
```

## Processing Modes

### Default: Sentence-Based Mode

**Advantages**:
- More natural speech flow
- Better prosody across sentence boundaries
- Fewer audio transitions
- More coherent meaning delivery

**Process**:
1. Group consecutive subtitle entries into complete sentences
2. Generate TTS audio for each complete sentence
3. Time sentences to start at the first subtitle's timestamp
4. Allow natural overflow with minimal gaps between sentences

### Alternative: Entry-Based Mode

**Advantages**:
- Closer adherence to original subtitle timing
- Easier debugging of timing issues
- More predictable behavior

**Process**:
1. Generate TTS audio for each individual subtitle entry
2. Maintain original subtitle timing as closely as possible
3. Handle multi-line entries as single utterances

## Example Workflow

### Input Files:
- Video: `01 Before Starting the Class.mp4` (28 minutes, 1677 seconds)
- Subtitles: `01.srt` (479 subtitle entries)

### Processing Steps:

1. **Parse SRT**: Extract 479 subtitle entries
2. **Group Sentences**: Combine into ~150 sentence groups (estimated)
3. **Generate TTS**: Create 150 audio files for complete sentences
4. **Assemble Audio**: Create 28-minute audio track with smart timing
5. **Process Video**: Replace original audio with English TTS

### Expected Results:
- Natural English narration following the pacing of the original
- Intelligent overflow handling maintaining overall timing
- High-quality 24kHz audio synchronized with video

## Timing Algorithm Deep Dive

### The Challenge

Original subtitle timing assumes the cadence of the source language (likely Korean/Japanese). English TTS may have different:
- Speaking rate
- Pause patterns
- Sentence length when spoken
- Emphasis and intonation timing

### The Solution: Flexible Timeline Assembly

**Core Principle**: Maintain overall video pacing while allowing natural speech flow.

**Algorithm Details**:

```python
class TimelineAssembler:
    def __init__(self, min_gap=0.250):
        self.current_time = 0.0
        self.min_gap = min_gap
        self.timeline = []

    def add_sentence(self, sentence_group, audio_duration):
        ideal_start = sentence_group.start_time

        # Calculate actual placement
        if self.current_time <= ideal_start:
            # We can start on time or early
            if self.current_time < ideal_start:
                # Insert silence to maintain timing
                silence_duration = ideal_start - self.current_time
                self.timeline.append(('silence', self.current_time, silence_duration))
            actual_start = ideal_start
        else:
            # We're running late - add minimum gap
            actual_start = self.current_time + self.min_gap

        # Place the audio
        self.timeline.append(('audio', actual_start, audio_duration, sentence_group))
        self.current_time = actual_start + audio_duration

        # Log timing deviation for analysis
        deviation = actual_start - ideal_start
        self.log_timing(sentence_group, deviation)
```

### Timing Scenarios Illustrated

**Scenario 1: Perfect Timing**
```
Subtitle 1: 00:00:00 -> 00:00:03 "Hello everyone."
Generated:  00:00:00 -> 00:00:02.5 (2.5s audio)
Result:     Audio at 00:00:00, silence 00:02:30 -> 00:03:00

Subtitle 2: 00:00:03 -> 00:00:06 "Welcome to the course."
Generated:  00:00:03 -> 00:00:05.8 (2.8s audio)
Result:     Audio starts exactly at 00:00:03
```

**Scenario 2: Minor Overflow**
```
Subtitle 1: 00:00:00 -> 00:00:03 "This is a longer sentence."
Generated:  00:00:00 -> 00:00:03.7 (3.7s audio, 0.7s overflow)
Result:     Audio plays 00:00:00 -> 00:00:03.7

Subtitle 2: 00:00:03 -> 00:00:06 "Next sentence."
Ideal:      Should start at 00:00:03
Actual:     Starts at 00:00:03.7 + 0.25 = 00:00:03.95
Generated:  00:00:03.95 -> 00:00:06.2 (2.25s audio)
```

**Scenario 3: Catch-Up Opportunity**
```
Previous audio ended at: 00:00:15.3
Next subtitle starts at: 00:00:18.0 (2.7s gap available)
Result: Insert 2.7s silence, start next audio exactly on time
```

## Error Handling and Robustness

### TTS Generation Errors
- **Fallback**: Skip problematic sentences, log errors
- **Recovery**: Continue processing remaining sentences
- **Reporting**: List failed sentences in final report

### Timing Edge Cases
- **Very Long Audio**: Cap maximum overflow at 10 seconds, insert forced break
- **Very Short Gaps**: Minimum 50ms between audio segments
- **Missing Subtitles**: Handle gaps gracefully with extended silence

### File Operation Errors
- **Permissions**: Check write access to working directory
- **Disk Space**: Estimate space requirements, warn if insufficient
- **Codec Issues**: Provide clear error messages for unsupported formats

## Performance Considerations

### TTS Generation
- **Caching**: Save generated audio files for reuse
- **Batching**: Process multiple sentences efficiently
- **Memory**: Stream large audio files rather than loading entirely

### Audio Processing
- **Streaming**: Process audio in chunks for large files
- **Format**: Use efficient audio formats during processing
- **Temporary Files**: Clean up intermediate files to save space

### Expected Processing Times
- **TTS Generation**: ~2-3 seconds per sentence (150 sentences = ~8 minutes)
- **Audio Assembly**: ~30 seconds for 28-minute final audio
- **Video Processing**: ~2 minutes for HD video
- **Total**: ~12 minutes for 28-minute input video

## Quality Control

### Audio Quality Metrics
- **Dynamic Range**: Ensure consistent volume levels
- **Clipping**: Prevent audio distortion
- **Sample Rate**: Maintain 24kHz throughout pipeline
- **Format Integrity**: Verify audio file headers and structure

### Timing Quality Metrics
- **Deviation Tracking**: Log timing differences from original subtitles
- **Gap Analysis**: Report silence durations and overflow amounts
- **Coverage**: Ensure complete audio coverage of video duration

### Output Validation
- **Duration Check**: Final audio duration matches video duration
- **Synchronization**: Spot-check timing alignment
- **Audio Continuity**: Verify no dropouts or artifacts

## Usage Examples

### Basic Usage
```bash
# Process a single video with default settings
python dubbing/main.py \
  --video "/mnt/d/Coloso/Syagamu/01 Before Starting the Class.mp4" \
  --srt "/mnt/d/Coloso/Syagamu/01.srt" \
  --working-dir "./working/01_before_starting"
```

### Advanced Usage
```bash
# Custom voice and timing settings
python dubbing/main.py \
  --video "video.mp4" \
  --srt "subtitles.srt" \
  --working-dir "./working/custom" \
  --voice "af_sky" \
  --min-gap 0.5 \
  --cleanup true
```

### Entry-by-Entry Mode
```bash
# Use original subtitle-by-subtitle approach
python dubbing/main.py \
  --video "video.mp4" \
  --srt "subtitles.srt" \
  --mode "entry" \
  --working-dir "./working/entry_mode"
```

### Debug Mode
```bash
# Keep all intermediate files and verbose logging
python dubbing/main.py \
  --video "video.mp4" \
  --srt "subtitles.srt" \
  --working-dir "./working/debug" \
  --cleanup false \
  --verbose true
```

## Implementation Checklist

### Phase 1: Core Parsing
- [ ] `srt_parser.py` - Basic SRT parsing
- [ ] `sentence_grouper.py` - Sentence detection algorithm
- [ ] Unit tests for parsing components

### Phase 2: TTS Integration
- [ ] `tts_generator.py` - Kokoro integration
- [ ] Audio file management and caching
- [ ] Error handling for TTS failures

### Phase 3: Audio Assembly
- [ ] `audio_assembler.py` - Timeline algorithm
- [ ] Silence insertion and gap handling
- [ ] Audio format standardization

### Phase 4: Video Processing
- [ ] `video_processor.py` - FFmpeg integration
- [ ] File management and cleanup
- [ ] Format compatibility testing

### Phase 5: Integration
- [ ] `main.py` - Complete workflow orchestration
- [ ] Command-line interface
- [ ] Configuration management
- [ ] Error reporting and logging

### Phase 6: Testing and Refinement
- [ ] End-to-end testing with sample videos
- [ ] Timing algorithm optimization
- [ ] Performance profiling and optimization
- [ ] Quality assurance and validation

This comprehensive system will transform subtitle-based videos into naturally-dubbed English versions, maintaining the educational value while making content accessible to English-speaking audiences.