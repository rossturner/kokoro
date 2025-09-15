#!/usr/bin/env python3
"""
Batch process all videos with embedded subtitles in the Mogoon course directory.
Skips videos that already have a corresponding output file.
"""

import os
import subprocess
import sys
from pathlib import Path
import time

# Configuration
SOURCE_DIR = Path("/mnt/d/Coloso/Mogoon - Fundamentals of Stylized Character Art")
BASE_OUTPUT_DIR = Path("./output")
CONDA_ACTIVATION = "/bin/bash -c 'source /home/ross/miniconda/etc/profile.d/conda.sh && conda activate kokoro'"
VOICE = "am_echo"  # Male American voice

def find_videos_with_embedded_subtitles(source_dir):
    """Find all video files with embedded subtitles in the source directory and subdirectories."""
    videos = []

    # Get all video files recursively
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv']
    for ext in video_extensions:
        # Use ** for recursive search in subdirectories
        videos.extend(source_dir.glob(f"**/*{ext}"))

    # Filter out any files in "Course Materials" directory
    videos = [v for v in videos if "Course Materials" not in str(v)]

    # Sort videos by path for consistent processing order
    return sorted(videos)

def check_output_exists(video_path, base_output_dir, source_dir):
    """Check if the output file already exists for a given video."""
    # Get relative path from source directory to maintain subdirectory structure
    relative_path = video_path.relative_to(source_dir)
    output_path = base_output_dir / "Mogoon - Fundamentals of Stylized Character Art" / relative_path
    return output_path.exists()

def process_video(video_path, verbose=False):
    """Process a single video with embedded subtitles."""
    # Build the command args
    cmd_str = f'source /home/ross/miniconda/etc/profile.d/conda.sh && conda activate kokoro && python -m dubbing.main --video "$1" --extract-subtitles --voice {VOICE}'
    if verbose:
        cmd_str += ' --verbose'

    args = [
        '/bin/bash', '-c',
        cmd_str,
        'bash',  # $0
        str(video_path)   # $1
    ]

    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"Subdirectory: {video_path.parent.name}")
    print(f"Video Path: {video_path}")
    print(f"Voice: {VOICE}")
    print(f"Extracting embedded subtitles: Yes")
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
    print("BATCH VIDEO DUBBING PROCESSOR - MOGOON COURSE")
    print("=" * 70)
    print(f"Source Directory: {SOURCE_DIR}")
    print(f"Output Directory: {BASE_OUTPUT_DIR}")
    print(f"Voice: {VOICE}")
    print(f"Embedded Subtitles: Yes")
    print()

    # Ensure output directory exists
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all videos with embedded subtitles
    print("Scanning for videos with embedded subtitles...")
    videos = find_videos_with_embedded_subtitles(SOURCE_DIR)

    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos across subdirectories")
    print()

    # Group videos by subdirectory for better organization
    by_subdir = {}
    for video in videos:
        subdir = video.parent.name
        if subdir not in by_subdir:
            by_subdir[subdir] = []
        by_subdir[subdir].append(video)

    # Show breakdown by subdirectory
    for subdir, subdir_videos in sorted(by_subdir.items()):
        print(f"  {subdir}: {len(subdir_videos)} videos")
    print()

    # Filter out already processed videos
    to_process = []
    skipped = []

    for video_path in videos:
        if check_output_exists(video_path, BASE_OUTPUT_DIR, SOURCE_DIR):
            skipped.append(video_path)
            print(f"⏭️  Skipping (already exists): {video_path.parent.name}/{video_path.name}")
        else:
            to_process.append(video_path)
            print(f"📋 To process: {video_path.parent.name}/{video_path.name}")

    print()
    print(f"Summary:")
    print(f"  - Total videos found: {len(videos)}")
    print(f"  - Already processed: {len(skipped)}")
    print(f"  - To be processed: {len(to_process)}")

    if not to_process:
        print("\n✨ All videos have already been processed!")
        return

    # Ask for confirmation
    print()
    try:
        response = input(f"Process {len(to_process)} videos with {VOICE} voice? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled by user")
            return
    except (EOFError, KeyboardInterrupt):
        print("Non-interactive environment detected or interrupted - exiting")
        return

    # Process each video
    successful = 0
    failed = 0

    for i, video_path in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] Processing {video_path.parent.name}/{video_path.name}...")

        if process_video(video_path):
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