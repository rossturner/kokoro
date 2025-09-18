# Kokoro TTS and Video Dubbing System

## Overview

This repository contains two main components:

1. **Kokoro TTS** - A high-quality Text-to-Speech system
2. **Video Dubbing System** - An intelligent video dubbing pipeline that uses Kokoro TTS

## Kokoro TTS

### What is Kokoro?

Kokoro is a neural text-to-speech system that generates natural-sounding speech from text input. It features multiple voices and supports real-time generation with high audio quality.

### Key Features

- **54 Voice Options**: Comprehensive voice library across multiple languages and styles (American, British, European, Asian)
- **High Quality**: 24kHz sample rate audio output
- **Real-time Generation**: Efficient streaming generation
- **Phoneme Support**: Advanced phonetic processing for natural speech
- **GPU Acceleration**: NVIDIA NVENC hardware acceleration for video encoding
- **Subtitle Extraction**: Extract embedded subtitle tracks from video files
- **Subtitle Alignment**: AI-powered subtitle timing correction using Whisper
- **Batch Processing**: Process multiple videos automatically
- **Advanced Compression**: Multiple quality presets (archive, balanced, compact)
- **Embedded Subtitles**: Burn-in subtitle tracks to output video

### Available Voices

**American Female (af_)**: `af_alloy`, `af_aoede`, `af_bella`, `af_heart` (default), `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`

**American Male (am_)**: `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

**British Female (bf_)**: `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`

**British Male (bm_)**: `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

**Other Languages**:
- **European**: `ef_dora`, `em_alex`, `em_santa`, `ff_siwis`
- **Hindi**: `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`
- **Italian**: `if_sara`, `im_nicola`
- **Japanese**: `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`
- **Portuguese**: `pf_dora`, `pm_alex`, `pm_santa`
- **Chinese**: `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`

### Basic Kokoro Usage

```python
from kokoro import KPipeline
import soundfile as sf

# Initialize pipeline
pipeline = KPipeline(lang_code='a')  # American English

# Generate speech
text = "Hello world! This is Kokoro TTS."
generator = pipeline(text, voice='af_heart')

# Process output
for i, (graphemes, phonemes, audio) in enumerate(generator):
    if audio is not None:
        audio_np = audio.numpy()
        sf.write(f'output_{i}.wav', audio_np, 24000)
```

### Environment Setup

```bash
# Create conda environment (if not already created)
conda env create -f environment.yml

# Activate conda environment
source /home/ross/miniconda/etc/profile.d/conda.sh
conda activate kokoro

# Verify activation
which python  # Should show: /home/ross/miniconda/envs/kokoro/bin/python
python --version  # Should show: Python 3.11.13

# Or use direct Python path (without activation)
/home/ross/miniconda/envs/kokoro/bin/python script.py
```

### Quick Start for Video Dubbing

```bash
# Navigate to project directory
cd /mnt/d/workspace/kokoro

# Activate conda environment
source /home/ross/miniconda/etc/profile.d/conda.sh
conda activate kokoro

# Run dubbing system
python -m dubbing.main \
  --video "/mnt/d/Coloso/Syagamu/01 Before Starting the Class.mp4" \
  --srt "/mnt/d/Coloso/Syagamu/01.srt" \
  --verbose
```

## Video Dubbing System

### What is the Video Dubbing System?

The Video Dubbing System is a comprehensive pipeline that converts subtitle-based videos into naturally dubbed English versions using Kokoro TTS. It features intelligent sentence grouping and timing algorithms that produce professional-quality results.

### Key Innovation: Sentence-Based Processing

Unlike simple subtitle-by-subtitle conversion, this system:

1. **Groups Subtitles into Sentences**: Combines consecutive subtitle entries that form complete thoughts
2. **Generates Natural Speech**: Creates TTS audio for complete sentences rather than fragments
3. **Smart Timing**: Allows audio to overflow subtitle boundaries while maintaining overall pacing
4. **Catch-up Logic**: Uses natural gaps in subtitles to realign timing

### System Architecture

```
Input: Video File (with or without SRT)
├── Subtitle Extractor: Extract embedded subtitles from video
├── SRT Parser: Extract and validate subtitles from SRT or extracted tracks
├── Sentence Grouper: Combine entries into complete sentences
├── TTS Generator: Create audio using Kokoro with 54 voice options
├── Audio Assembler: Build timeline with intelligent timing and normalization
├── Subtitle Aligner: AI-powered subtitle timing correction using Whisper
└── Video Processor: GPU-accelerated encoding with compression presets
    ├── Strip original audio track
    ├── Add new TTS audio track
    ├── Embed aligned subtitles (optional)
    └── Apply compression settings (archive/balanced/compact)
```

### Example: Sentence Grouping

**Input Subtitles:**
```
8. (30.505-34.189) "I work as a freelance illustrator mainly doing"
9. (34.189-37.874) "character-centered illustrations and character"
10. (37.874-42.441) "designs for casual games and virtual YouTuber industries."
```

**System Output:**
- **Single Sentence**: "I work as a freelance illustrator mainly doing character-centered illustrations and character designs for casual games and virtual YouTuber industries."
- **Timing**: Starts at 30.505s, may extend beyond 42.441s if needed
- **Next Audio**: Waits for slight gap then continues with next sentence

### Usage Examples

#### Basic Usage
```bash
# Process single video with defaults (auto-extracts embedded subtitles)
python -m dubbing.main \
  --video "/path/to/video.mp4"

# Process with external SRT file
python -m dubbing.main \
  --video "/path/to/video.mp4" \
  --srt "/path/to/subtitles.srt"
```

#### Advanced Usage
```bash
# Custom voice, GPU acceleration, and compression
python -m dubbing.main \
  --video "/mnt/d/Coloso/Syagamu/01 Before Starting the Class.mp4" \
  --srt "/mnt/d/Coloso/Syagamu/01.srt" \
  --voice af_heart \
  --output-dir "./final_output" \
  --preset balanced \
  --gpu \
  --embed-subtitles \
  --align-subtitles \
  --verbose \
  --cleanup
```

#### Batch Processing
```bash
# Process all videos in a directory automatically
python batch_process.py

# The script will:
# - Find all video files in /mnt/d/Coloso/Syagamu/
# - Match them with corresponding SRT files
# - Skip videos that are already processed
# - Process remaining videos with consistent settings
```

#### Configuration Options
```bash
# Voice Selection (54 options available)
--voice af_heart     # Default American female voice
--voice am_adam      # American male voice
--voice bf_alice     # British female voice
--voice jf_alpha     # Japanese female voice
--voice zm_yunxi     # Chinese male voice

# Quality Presets
--preset archive     # High quality (CRF 18, slow preset, 192k audio)
--preset balanced    # Default (CRF 23, medium preset, 128k audio)
--preset compact     # Smaller files (CRF 28, fast preset, 96k audio)

# GPU Acceleration
--gpu                # Enable NVIDIA NVENC acceleration
--gpu-preset p4      # GPU encoding preset (p1-p7, p1=fastest, p7=best)

# Subtitle Options
--extract-subtitles  # Extract embedded subtitle tracks from video
--align-subtitles    # AI-powered subtitle timing correction
--embed-subtitles    # Burn-in English subtitles to output video

# Processing modes
--mode sentence      # Default: intelligent sentence grouping
--mode entry         # Alternative: subtitle-by-subtitle

# Timing control
--min-gap 0.25       # Minimum gap between overflowing sentences (250ms)
```

### File Organization

The system maintains clean organization:

```
working/[video_name]/
├── video.mp4              # Copied original video
├── subtitles.srt          # Copied SRT file
├── sentences.json         # Parsed sentence groups (debug)
├── audio_snippets/        # Individual TTS audio files
│   ├── sentence_001.wav
│   ├── sentence_002.wav
│   └── ...
├── final_audio.wav        # Combined audio track
├── timeline_debug.json    # Timing analysis (debug)
├── output_video.mp4       # Final dubbed video
└── pipeline_stats.json    # Processing statistics
```

### Timing Algorithm

The core innovation is the intelligent timing system:

```python
current_time = 0.0
min_gap = 0.250  # 250ms minimum gap

for sentence in sentences:
    ideal_start = sentence.start_time

    if current_time <= ideal_start:
        # Can start on time - insert silence if needed
        actual_start = ideal_start
        if current_time < ideal_start:
            insert_silence(ideal_start - current_time)
    else:
        # Running late - add minimum gap
        actual_start = current_time + min_gap

    # Place sentence audio
    place_audio(sentence, at=actual_start)
    current_time = actual_start + audio_duration
```

This allows:
- **Natural Speech Flow**: Complete sentences sound more fluent
- **Flexible Timing**: Audio can extend past subtitle boundaries
- **Automatic Catch-up**: Gaps in subtitles allow timeline realignment
- **Minimal Interruption**: Only adds gaps when absolutely necessary

### Processing Pipeline Phases

1. **File Setup** (30s)
   - Copy video to working directory
   - Validate input files and extract video info
   - Detect and extract embedded subtitle tracks (if present)

2. **Subtitle Processing** (10-30s)
   - Extract embedded subtitles or parse SRT file
   - Extract ~479 subtitle entries from source
   - Skip instruction entries ("turn off subtitles")
   - Validate timing and content

3. **Group Sentences** (5s)
   - Analyze subtitle text for sentence boundaries
   - Combine ~479 entries into ~150 complete sentences
   - Save grouping analysis for debugging

4. **Generate TTS Audio** (8-12 minutes)
   - Initialize Kokoro TTS pipeline with selected voice
   - Generate audio for each complete sentence
   - Apply audio normalization and quality control
   - Save individual audio files (sentence_001.wav, etc.)

5. **Assemble Audio** (30s)
   - Apply intelligent timing algorithm
   - Insert silence where needed for catch-up
   - Normalize and balance audio levels
   - Create final 28-minute audio track

6. **Subtitle Alignment** (1-3 minutes, optional)
   - Use Whisper to transcribe generated TTS audio
   - Align subtitle timing with actual speech
   - Generate corrected subtitle track

7. **Create Final Video** (30s-5 minutes)
   - Strip original audio from video
   - Replace with English TTS audio track
   - Embed aligned subtitles (optional)
   - Apply GPU-accelerated compression with selected preset
   - Export final dubbed video

**Total Processing Time**:
- CPU-only: ~12-15 minutes for 28-minute video
- GPU-accelerated: ~8-10 minutes for 28-minute video

### Quality Features

- **Audio Quality**: 24kHz output, advanced normalization, dynamic range optimization, no clipping
- **Video Quality**: Multiple compression presets (archive/balanced/compact) with GPU acceleration
- **Timing Accuracy**: AI-powered subtitle alignment with Whisper for precise synchronization
- **Error Recovery**: Graceful TTS failure handling with detailed fallback mechanisms
- **Validation**: Comprehensive video-audio synchronization checks and duration validation
- **Statistics**: Detailed reporting on processing success rates, timing accuracy, and performance metrics
- **Subtitle Support**: Extract embedded tracks, align timing, and burn-in corrected subtitles
- **Performance**: GPU acceleration reduces processing time by 40-60%

### Real-world Example

Processing "01 Before Starting the Class.mp4" (28 minutes, 479 subtitles):

**Input**: Korean/Japanese educational video with English subtitles
**Output**: Naturally dubbed English video with `af_heart` voice

**Results**:
- Original: 479 subtitle entries
- Grouped: ~150 complete sentences
- Success Rate: >95% TTS generation success
- Timing: <1 second average deviation from original pacing
- Quality: Professional-sounding English narration

### Dependencies

**Core Requirements**:
- Python 3.11+
- Kokoro TTS package (>=0.9.4)
- soundfile (audio I/O)
- numpy (audio processing)
- faster-whisper (subtitle alignment)
- misaki[en] (enhanced text processing)

**AI/ML Frameworks**:
- PyTorch with CUDA 12.1 support (GPU acceleration)
- torchvision and torchaudio

**Video Processing**:
- FFmpeg with NVENC support (video/audio manipulation)
- ffprobe (video analysis)

**Optional**:
- conda (environment management)
- NVIDIA GPU with NVENC support (hardware acceleration)

### Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate kokoro

# Or install directly
pip install kokoro>=0.9.4 soundfile

# Ensure FFmpeg is available
ffmpeg -version
```

### Testing

Each module includes test functionality:

```bash
# Test individual components
python dubbing/srt_parser.py      # Test SRT parsing
python dubbing/sentence_grouper.py # Test sentence grouping
python dubbing/tts_generator.py   # Test TTS generation
python dubbing/audio_assembler.py # Test audio assembly
python dubbing/video_processor.py # Test video processing

# Test complete pipeline
python dubbing/main.py --video "test.mp4" --srt "test.srt" --verbose
```

### Troubleshooting

**Common Issues**:

1. **"FFmpeg not found"**
   - Install FFmpeg: `apt install ffmpeg` or `brew install ffmpeg`

2. **"Pipeline not initialized"**
   - Ensure Kokoro environment: `conda activate kokoro`
   - Check dependencies: `pip list | grep kokoro`

3. **"High TTS failure rate"**
   - Check text encoding in SRT file
   - Verify Kokoro voice availability
   - Try different voice: `--voice af_sky`

4. **"Timing issues"**
   - Adjust minimum gap: `--min-gap 0.5`
   - Use entry mode: `--mode entry`
   - Check subtitle timing accuracy

### Advanced Configuration

Create custom configuration file:

```json
{
    "voice": "af_heart",
    "sample_rate": 24000,
    "min_sentence_gap": 0.25,
    "sentence_mode": true,
    "cleanup_intermediate_files": false,
    "verbose_logging": true
}
```

Use with: `--config config.json`

### Performance Optimization

For faster processing:
- Use SSD storage for working directory
- Ensure sufficient RAM (4GB+ recommended)
- Close other applications during TTS generation
- Consider processing shorter segments for very long videos

### Output Quality

The system produces:
- **Natural Speech**: Complete sentences with proper intonation
- **Consistent Pacing**: Maintains educational video rhythm
- **High Fidelity**: 24kHz audio quality matching professional standards
- **Synchronized Timing**: Proper video-audio alignment
- **Clean Output**: No artifacts or audio dropouts

This system transforms subtitle-based educational content into accessible, professionally dubbed English videos while preserving the original timing and educational value.

## Advanced Features and Modular Architecture

### Core Modules

The system is built with a modular architecture for maintainability and extensibility:

**Core Processing Modules**:
- `srt_parser.py`: Subtitle parsing and validation with encoding detection
- `sentence_grouper.py`: Intelligent sentence boundary detection and grouping
- `tts_generator.py`: Kokoro TTS integration with 82 voice options
- `audio_assembler.py`: Advanced audio timeline assembly with normalization
- `video_processor.py`: GPU-accelerated video processing with compression presets

**Enhanced Processing Modules**:
- `subtitle_extractor.py`: Extract embedded subtitle tracks from video files using FFmpeg
- `subtitle_aligner.py`: AI-powered subtitle timing correction using faster-whisper
- `config.py`: Comprehensive configuration system with validation and presets
- `constants.py`: Centralized configuration constants and thresholds
- `utils.py`: Shared utility functions for audio processing and validation

**Batch Processing**:
- `batch_process.py`: Automated processing of entire video directories
- Smart file matching (handles various naming patterns: "01.srt" ↔ "01 Video Title.mp4")
- Skip already-processed files to resume interrupted batch jobs

### Configuration System

The enhanced configuration system provides extensive customization:

```python
from dubbing.config import Config, get_default_config

# Get default configuration
config = get_default_config()

# Customize settings
config.voice = 'af_heart'
config.enable_gpu_acceleration = True
config.video_crf = 23  # Compression quality
config.embed_subtitles = True
config.align_subtitles = True

# Apply preset
config.apply_compression_preset('balanced')  # or 'archive', 'compact'
config.apply_gpu_preset('gpu_balanced')      # GPU-optimized settings

# Validate configuration
issues = config.validate()
if not issues:
    print("Configuration is valid")
```

### GPU Acceleration

NVIDIA NVENC hardware acceleration dramatically improves processing speed:

**Benefits**:
- **40-60% faster video encoding** compared to CPU-only processing
- **Lower system resource usage** during video processing
- **Consistent quality** with dedicated video encoding hardware

**Requirements**:
- NVIDIA GPU with NVENC support (GTX 1660+, RTX series, or professional cards)
- Recent GPU drivers with NVENC support
- FFmpeg compiled with NVENC support

**GPU Presets**:
```bash
--preset gpu_archive   # High quality GPU encoding (CQ 18, preset p6)
--preset gpu_balanced  # Balanced GPU encoding (CQ 23, preset p4) [Default]
--preset gpu_fast      # Fast GPU encoding (CQ 26, preset p2)
```

### Subtitle Alignment with Whisper

The subtitle alignment feature uses OpenAI's Whisper to improve timing accuracy:

**How it works**:
1. **TTS Audio Generation**: Generate dubbed audio using Kokoro TTS
2. **Whisper Transcription**: Transcribe the generated audio to get word-level timestamps
3. **Alignment Matching**: Match original subtitle text with Whisper transcription
4. **Timing Correction**: Adjust subtitle timing to match actual speech patterns

**Benefits**:
- **Improved synchronization**: Subtitles match actual speech timing
- **Better readability**: Subtitles appear and disappear at natural speech boundaries
- **Professional quality**: Eliminates timing drift common in TTS systems

**Configuration**:
```bash
--align-subtitles           # Enable subtitle alignment
--whisper-model large-v3    # Whisper model size (base, small, medium, large-v3)
```

### Batch Processing Capabilities

Process entire video libraries automatically:

```bash
python batch_process.py
```

**Features**:
- **Smart File Matching**: Automatically pairs videos with subtitle files
- **Flexible Naming**: Handles multiple naming patterns (01.srt ↔ 01 Title.mp4)
- **Resume Support**: Skips already-processed videos
- **Progress Tracking**: Detailed logging of batch processing progress
- **Error Recovery**: Continues processing if individual videos fail

**Supported Patterns**:
- `01.srt` ↔ `01 Before Starting the Class.mp4`
- `03-1.srt` ↔ `03-1 Setting up Clip Studio Paint.mp4`
- `video.srt` ↔ `video.mp4` (exact name match)

### Performance Optimizations

**Audio Processing**:
- Advanced normalization algorithms prevent clipping and optimize dynamic range
- Soft limiting ensures consistent volume levels across all generated speech
- Quiet audio boost improves intelligibility of low-volume segments

**Video Processing**:
- GPU acceleration reduces encoding time by 40-60%
- Smart preset selection balances quality and file size
- Parallel processing where possible (file I/O, validation, etc.)

**Memory Management**:
- Chunked audio processing for large files
- Efficient subtitle parsing with encoding detection
- Cleanup of intermediate files (optional)

This comprehensive system provides professional-quality video dubbing with extensive customization options and optimizations for both quality and performance.