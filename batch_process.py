#!/usr/bin/env python3
"""
Batch process all video-subtitle pairs in the Syagamu directory.
Skips videos that already have a corresponding output file.
"""

import os
import subprocess
import sys
from pathlib import Path
import time

# Configuration
SOURCE_DIR = Path("/mnt/d/Coloso/Syagamu")
BASE_OUTPUT_DIR = Path("./output")
CONDA_ACTIVATION = "/bin/bash -c 'source /home/ross/miniconda/etc/profile.d/conda.sh && conda activate kokoro'"

def find_video_subtitle_pairs(source_dir):
    """Find all matching video-subtitle pairs in the source directory."""
    pairs = []

    # Get all video files
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(source_dir.glob(f"*{ext}"))

    # For each video, check if there's a matching SRT file
    for video_path in sorted(video_files):
        # Get the base name without extension
        base_name = video_path.stem

        # Look for matching SRT file with multiple naming patterns
        srt_candidates = [
            source_dir / f"{base_name}.srt",
        ]

        # Extract number prefix patterns for different video naming schemes
        if base_name[:2].isdigit():
            # Pattern 1: "01.srt" for "01 Before Starting the Class.mp4"
            number_prefix = base_name[:2]
            srt_candidates.append(source_dir / f"{number_prefix}.srt")

            # Pattern 2: "03-1.srt" for "03-1 Setting up Clip Studio Paint..."
            # Look for dash after the initial digits
            dash_pos = base_name.find('-')
            if dash_pos >= 2 and dash_pos < len(base_name) - 1:
                # Check if there's a digit after the dash
                next_char_pos = dash_pos + 1
                if next_char_pos < len(base_name) and base_name[next_char_pos].isdigit():
                    # Extract up to the next space or end of digits
                    extended_prefix = base_name[:dash_pos+1]  # "03-"
                    i = next_char_pos
                    while i < len(base_name) and base_name[i].isdigit():
                        extended_prefix += base_name[i]
                        i += 1
                    srt_candidates.append(source_dir / f"{extended_prefix}.srt")

        # Find the first existing SRT file
        srt_path = None
        for candidate in srt_candidates:
            if candidate.exists():
                srt_path = candidate
                break

        if srt_path:
            pairs.append((video_path, srt_path))
        else:
            print(f"⚠️  No subtitle file found for: {video_path.name}")

    return pairs

def check_output_exists(video_path, base_output_dir):
    """Check if the output file already exists for a given video."""
    output_filename = video_path.name
    source_dir_name = video_path.parent.name
    output_path = base_output_dir / source_dir_name / output_filename
    return output_path.exists()

def process_video(video_path, srt_path, verbose=False):
    """Process a single video-subtitle pair."""
    # Build the command args
    args = [
        '/bin/bash', '-c',
        'source /home/ross/miniconda/etc/profile.d/conda.sh && conda activate kokoro && python -m dubbing.main --video "$1" --srt "$2"' + (' --verbose' if verbose else ''),
        'bash',  # $0
        str(video_path),  # $1
        str(srt_path)     # $2
    ]

    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"Subtitles: {srt_path.name}")
    print(f"Video Path: {video_path}")
    print(f"SRT Path: {srt_path}")
    print(f"{'='*60}\n")

    # Execute the command
    start_time = time.time()
    try:
        result = subprocess.run(args, capture_output=False, text=True)
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Successfully processed {video_path.name} in {elapsed_time:.1f} seconds")
            return True
        else:
            print(f"❌ Failed to process {video_path.name}")
            return False
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error processing {video_path.name}: {e}")
        return False

def main():
    """Main execution function."""
    print("=" * 70)
    print("BATCH VIDEO DUBBING PROCESSOR")
    print("=" * 70)
    print(f"Source Directory: {SOURCE_DIR}")
    print(f"Output Directory: {BASE_OUTPUT_DIR}")
    print()

    # Ensure output directory exists
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all video-subtitle pairs
    print("Scanning for video-subtitle pairs...")
    pairs = find_video_subtitle_pairs(SOURCE_DIR)

    if not pairs:
        print("No video-subtitle pairs found!")
        return

    print(f"Found {len(pairs)} video-subtitle pairs")
    print()

    # Filter out already processed videos
    to_process = []
    skipped = []

    for video_path, srt_path in pairs:
        if check_output_exists(video_path, BASE_OUTPUT_DIR):
            skipped.append(video_path.name)
            print(f"⏭️  Skipping (already exists): {video_path.name}")
        else:
            to_process.append((video_path, srt_path))
            print(f"📋 To process: {video_path.name}")

    print()
    print(f"Summary:")
    print(f"  - Total pairs found: {len(pairs)}")
    print(f"  - Already processed: {len(skipped)}")
    print(f"  - To be processed: {len(to_process)}")

    if not to_process:
        print("\n✨ All videos have already been processed!")
        return

    # Ask for confirmation
    print()
    try:
        response = input(f"Process {len(to_process)} videos? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled by user")
            return
    except (EOFError, KeyboardInterrupt):
        print("Non-interactive environment detected or interrupted - exiting")
        return

    # Process each video
    successful = 0
    failed = 0

    for i, (video_path, srt_path) in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] Processing {video_path.name}...")

        if process_video(video_path, srt_path):
            successful += 1
        else:
            failed += 1
            # Ask if user wants to continue after a failure
            if i < len(to_process):
                try:
                    response = input("Continue with remaining videos? (y/n): ").strip().lower()
                    if response != 'y':
                        print("Stopped by user")
                        break
                except (EOFError, KeyboardInterrupt):
                    print("Non-interactive environment - stopping after failure")
                    break

    # Final summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped: {len(skipped)}")

    if failed > 0:
        print("\nFailed videos can be retried by running the script again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)