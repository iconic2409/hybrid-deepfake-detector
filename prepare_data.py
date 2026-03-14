#!/usr/bin/env python3
"""
Prepare Deepfake Dataset

This script extracts frames from videos and saves them into:
data/
    train/
        real/
        fake/
    val/
        real/
        fake/

Frames will be resized to 224x224 for compatibility with ResNet-18.
"""

import os
import cv2
from pathlib import Path
import argparse
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_frames(video_path, output_dir, max_frames=None, frame_skip=5):
    """
    Extract frames from a video and save them as images.

    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save frames
        max_frames (int, optional): Maximum number of frames to extract
        frame_skip (int): Extract every 'frame_skip' frame
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            output_file = output_dir / f"{video_path.stem}_frame{saved_count:04d}.jpg"
            cv2.imwrite(str(output_file), frame_resized)
            saved_count += 1
            if max_frames and saved_count >= max_frames:
                break
        frame_count += 1

    cap.release()
    logger.info(f"Extracted {saved_count} frames from {video_path}")
    return saved_count

def process_videos(input_dir, output_dir, max_frames_per_video=None, frame_skip=5):
    """
    Process all videos in a directory.

    Args:
        input_dir (str): Folder containing videos
        output_dir (str): Folder to save frames
        max_frames_per_video (int): Max frames to extract per video
        frame_skip (int): Extract every n-th frame
    """
    input_dir = Path(input_dir)
    video_files = list(input_dir.rglob("*.mp4")) + list(input_dir.rglob("*.mov")) + list(input_dir.rglob("*.avi"))
    logger.info(f"Found {len(video_files)} videos in {input_dir}")

    total_frames = 0
    for video_file in video_files:
        frames_saved = extract_frames(video_file, Path(output_dir) / video_file.stem,
                                      max_frames=max_frames_per_video, frame_skip=frame_skip)
        total_frames += frames_saved
    logger.info(f"Total frames extracted: {total_frames}")

def split_data(source_dir, dest_dir, split_ratio=0.8):
    """
    Split extracted frames into train and validation sets.

    Args:
        source_dir (str): Folder containing real/ and fake/ subfolders
        dest_dir (str): Folder to save train/ and val/ splits
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    for class_name in ["real", "fake"]:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            continue
        images = list(class_dir.rglob("*.jpg"))
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)

        train_dir = dest_dir / "train" / class_name
        val_dir = dest_dir / "val" / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images[:split_idx]:
            dest_path = train_dir / img_path.name
            os.rename(str(img_path), str(dest_path))
        for img_path in images[split_idx:]:
            dest_path = val_dir / img_path.name
            os.rename(str(img_path), str(dest_path))
        logger.info(f"{class_name}: {len(images[:split_idx])} train, {len(images[split_idx:])} val images")

def main():
    parser = argparse.ArgumentParser(description="Prepare deepfake dataset from videos")
    parser.add_argument("--real_videos", type=str, required=True, help="Path to real videos folder")
    parser.add_argument("--fake_videos", type=str, required=True, help="Path to fake videos folder")
    parser.add_argument("--output_dir", type=str, default="data", help="Output folder for frames")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames per video")
    parser.add_argument("--frame_skip", type=int, default=5, help="Extract every n-th frame")
    args = parser.parse_args()

    # Extract frames
    process_videos(args.real_videos, Path(args.output_dir) / "real", max_frames_per_video=args.max_frames, frame_skip=args.frame_skip)
    process_videos(args.fake_videos, Path(args.output_dir) / "fake", max_frames_per_video=args.max_frames, frame_skip=args.frame_skip)

    # Split into train/val
    split_data(Path(args.output_dir), Path(args.output_dir), split_ratio=0.8)
    logger.info("Dataset preparation completed!")

if __name__ == "__main__":
    main()
