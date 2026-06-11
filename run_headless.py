#!/usr/bin/env python3
"""
EcoMotionZip — Headless / Command-Line Mode
============================================
Runs the compression pipeline without a graphical interface.
Designed for Raspberry Pi and remote/server environments.

Usage
-----
    python run_headless.py --video_source /path/to/video.mp4 --output_directory /path/to/output

    Or use config.json for defaults and override individual values:

    python run_headless.py --movement_threshold 30 --video_codec X264

Run with --help to see all available options:

    python run_headless.py --help

Requirements
------------
    pip install -r requirements.txt
"""
from ecomotioinzip.pipeline import read_args, main as run_pipeline

if __name__ == "__main__":
    cfg = read_args()
    run_pipeline(cfg)
