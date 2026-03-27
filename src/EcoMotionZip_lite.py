"""
EcoMotionZip — Motion-based video compression pipeline.

This script processes video files (or a live camera feed) and produces a
compressed output that retains only the frames where motion is detected.
Stationary background frames are discarded, dramatically reducing file size
for footage with intermittent activity (e.g. wildlife cameras, security feeds).

Pipeline overview
-----------------
Three threads run concurrently and communicate via bounded queues:

    Reader  ──(reading_queue)──►  MotionDetector  ──(writing_queue)──►  Writer

1. **Reader** decodes frames from a video file or camera and pushes them onto
   the reading_queue. For video files it uses "smart sleeping" to avoid
   flooding the queue faster than downstream threads can consume it.

2. **MotionDetector** pulls frames from the reading_queue, computes a
   per-pixel motion mask by differencing consecutive frames, and forwards
   only the frames (and their masks) that contain motion onto the
   writing_queue. A configurable "buffer" keeps recording for a few extra
   frames after motion stops to capture the tail of an event.

3. **Writer** pulls motion frames from the writing_queue and encodes them to
   disk using either OpenCV's VideoWriter (default) or an FFmpeg subprocess
   (fallback for codecs like H.265/HEVC that OpenCV cannot encode). It also
   writes a CSV sidecar that maps each output frame number back to its
   original frame number, and optionally saves individual JPEG snapshots.

Configuration is loaded from ``config.json`` at the project root. Any field
can be overridden at runtime with the equivalent ``--flag`` CLI argument.

Usage
-----
    python EcoMotionZip_lite.py
    python EcoMotionZip_lite.py --video_source /path/to/video.mp4
    python EcoMotionZip_lite.py --movement_threshold 30 --downscale_factor 8
"""

from __future__ import annotations
import os
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, List, Optional, Tuple, Union
from itertools import product
import argparse
import cv2
import numpy as np
import subprocess

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class Config:
    """Holds all runtime parameters for a single processing run.

    An instance of this class is constructed from ``config.json`` and any
    CLI overrides, then passed to ``main()``. All three pipeline threads
    (Reader, MotionDetector, Writer) draw their settings from this object.

    Attributes
    ----------
    video_source : str or int
        Path to an input video file or directory of videos.  An integer
        value selects a live camera by device index (0 = default webcam).
    output_directory : str
        Root directory for output. A sub-folder ``EcoMotionZip/<filename>/``
        is created automatically inside this directory.
    record_duration : int
        Maximum recording time in seconds when reading from a live camera.
        Has no effect when processing pre-recorded video files.
    number_of_videos : int
        Number of consecutive recordings to capture from a live camera.
    delete_original_after_processing : bool
        If True, the input video file is deleted once processing completes
        successfully.  Has no effect on camera sources.
    embed_timestamps : bool
        If True, a text overlay showing the compressed frame number, original
        frame number and wall-clock time is burned into each output frame.
    camera_resolution : Tuple[int, int]
        ``(width, height)`` in pixels for live camera capture.  Ignored when
        reading from a file.
    camera_fps : int
        Target frame rate for live camera capture.  Ignored when reading from
        a file.
    raspberrypi_camera : bool
        If True, use the ``picamera2`` library to access a Raspberry Pi
        camera module instead of OpenCV's VideoCapture.
    reader_sleep_seconds : float
        Initial sleep duration (seconds) used by the Reader when the reading
        queue is full.  The smart-sleep algorithm adjusts this dynamically.
    reader_flush_proportion : float
        Queue fill fraction (0–1) that triggers the Reader's sleep.  A value
        of 0.9 means the Reader sleeps when the queue is 90 % full.
    downscale_factor : int
        The frame is downscaled by this factor before computing the motion
        mask.  Higher values are faster but less sensitive to fine movement.
        A value of 1 disables downscaling.
    dilate_kernel_size : int
        Size of the square morphological dilation kernel applied to the motion
        mask.  Larger values merge nearby motion blobs and fill small holes.
        This is automatically scaled down by ``downscale_factor``.
    movement_threshold : int
        Pixel-intensity difference (0–255) that a pixel must exceed to be
        counted as moving.  Lower values detect subtle motion; higher values
        ignore small variations (e.g. camera noise, gentle swaying leaves).
    post_motion_record_frames : float
        Number of additional frames to capture after motion stops.  Useful
        for catching the tail end of an event (e.g. an animal walking out of
        frame).  Set to 0 to disable.
    full_frame_capture_interval : int
        Every N frames, one fully unmasked keyframe is recorded (regardless
        of whether it is the start of a motion sequence).  This provides
        spatial context in the compressed video and aids downstream analysis.
    video_codec : str
        Four-character code (FOURCC) or alias for the output codec.
        Supported values: ``DIVX``, ``X264``, ``FFV1``, ``HEVC``, ``H265``.
        HEVC is encoded via an FFmpeg subprocess; all others use OpenCV.
    num_opencv_threads : int
        Maximum number of threads OpenCV is allowed to use internally.
        Reducing this avoids CPU contention with the pipeline's own threads.
    background_transparency : float
        Controls how much of the stationary background is blended into the
        output for non-keyframe motion frames (0.0–1.0).  At 0.0 the
        background is fully black; at 1.0 the full original frame is shown
        (effectively no compression of pixel values).
    save_frames : bool
        If True, individual motion frames are saved as JPEG files alongside
        the output video.
    frames_to_save : int
        Maximum number of JPEG frames to save.  Saving stops once this limit
        is reached, even if more motion frames exist.
    """

    def __init__(
        self,
        video_source: str,
        output_directory: str,
        record_duration: int,
        number_of_videos: int,
        delete_original_after_processing: bool,
        embed_timestamps: bool,
        camera_resolution: Tuple[int, int],
        camera_fps: int,
        raspberrypi_camera: bool,
        reader_sleep_seconds: float,
        reader_flush_proportion: float,
        downscale_factor: int,
        dilate_kernel_size: int,
        movement_threshold: int,
        post_motion_record_frames: float,
        full_frame_capture_interval: int,
        video_codec: str,
        num_opencv_threads: int,
        background_transparency: float,
        save_frames: bool,
        frames_to_save: int,
    ) -> None:
        self.video_source = video_source
        self.output_directory = output_directory
        self.record_duration = record_duration
        self.number_of_videos = number_of_videos
        self.camera_resolution = camera_resolution
        self.camera_fps = camera_fps
        self.raspberrypi_camera = raspberrypi_camera
        self.delete_original_after_processing = delete_original_after_processing
        self.embed_timestamps = embed_timestamps
        self.reader_sleep_seconds = reader_sleep_seconds
        self.reader_flush_proportion = reader_flush_proportion
        self.downscale_factor = downscale_factor
        self.dilate_kernel_size = dilate_kernel_size
        self.movement_threshold = movement_threshold
        self.post_motion_record_frames = post_motion_record_frames
        self.full_frame_capture_interval = full_frame_capture_interval
        self.video_codec = video_codec
        self.num_opencv_threads = num_opencv_threads
        self.background_transparency = background_transparency
        self.save_frames = save_frames
        self.frames_to_save = frames_to_save


def read_args():
    """Parse CLI arguments and return them as a flat dictionary.

    Only arguments explicitly passed on the command line are returned with
    non-None values.  The caller is responsible for merging these values with
    the defaults loaded from ``config.json`` — CLI arguments take precedence
    over the JSON file.

    Returns
    -------
    dict
        Keys match the field names of ``Config``.  Any argument not supplied
        on the command line has a value of ``None`` and should be ignored
        during the merge.
    """
    parser = argparse.ArgumentParser(
        description="EcoMotionZip — compress video by retaining only motion frames."
    )
    parser.add_argument(
        "--video_source",
        type=lambda x: int(x) if x.isdigit() else str(x),
        help=(
            "Path to an input video file or directory of videos.  "
            "Pass an integer (e.g. 0) to read from a live camera instead."
        ),
    )
    parser.add_argument("--output_directory", type=str, help="Root directory for output files.")
    parser.add_argument(
        "--record_duration",
        type=int,
        help="Maximum recording time per video in seconds (live camera only).",
    )
    parser.add_argument(
        "--number_of_videos",
        type=int,
        help="Number of consecutive recordings to capture (live camera only).",
    )
    parser.add_argument(
        "--camera_resolution",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        help="Live camera resolution as width and height, e.g. --camera_resolution 1920 1080.",
    )
    parser.add_argument("--camera_fps", type=int, help="Target frame rate for live camera capture.")
    parser.add_argument(
        "--delete_original_after_processing",
        type=bool,
        help="Delete the input video file after successful processing.",
    )
    parser.add_argument(
        "--downscale_factor",
        type=int,
        help=(
            "Factor by which frames are downscaled before computing the motion mask.  "
            "Higher = faster but less sensitive.  1 = no downscaling."
        ),
    )
    parser.add_argument(
        "--dilate_kernel_size",
        type=int,
        help="Size of the dilation kernel applied to the motion mask (in original-resolution pixels).",
    )
    parser.add_argument(
        "--movement_threshold",
        type=int,
        help="Minimum per-pixel intensity difference (0–255) required to flag a pixel as moving.",
    )
    parser.add_argument(
        "--post_motion_record_frames",
        type=float,
        help="Extra frames to record after motion stops, to capture the tail of an event.",
    )
    parser.add_argument(
        "--full_frame_capture_interval",
        type=int,
        help="Interval (in input frames) between unmasked keyframe captures.",
    )
    parser.add_argument(
        "--video_codec",
        type=str,
        choices=["DIVX", "X264", "FFV1", "HEVC", "H265"],
        help="Output video codec.  HEVC is encoded via FFmpeg; all others use OpenCV.",
    )
    parser.add_argument(
        "--num_opencv_threads",
        type=int,
        help="Maximum OpenCV internal threads.  Lower values reduce CPU contention.",
    )
    parser.add_argument(
        "--background_transparency",
        type=float,
        help=(
            "Background blend level for non-keyframe motion frames (0.0–1.0).  "
            "0 = black background, 1 = full original frame visible."
        ),
    )
    parser.add_argument("--save_frames", type=bool, help="Save individual motion frames as JPEG.")
    parser.add_argument(
        "--frames_to_save",
        type=int,
        help="Maximum number of JPEG frames to save alongside the output video.",
    )

    args = parser.parse_args()
    return vars(args)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
# config.json is located at the project root (one level above this file's
# directory).  Using an absolute path derived from __file__ makes the script
# work regardless of the current working directory.
_CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(_CONFIG_PATH, "r") as _f:
    __config_dict = json.load(_f)

# CLI arguments override JSON defaults.  Only non-None CLI values are merged
# so that unspecified flags don't silently zero-out JSON fields.
cmd_args = read_args()
__config_dict.update((k, v) for k, v in cmd_args.items() if v is not None)
CONFIG = Config(**__config_dict)


# ---------------------------------------------------------------------------
# Base thread class
# ---------------------------------------------------------------------------

class LoggingThread(Thread):
    """``threading.Thread`` subclass that exposes a per-thread logger.

    All three pipeline threads (Reader, MotionDetector, Writer) inherit from
    this class.  Wrapping the logger calls here means each thread can call
    ``self.info(...)`` directly without passing the logger around, and every
    log message is automatically tagged with the thread's name by the
    formatter configured in ``main()``.

    Attributes
    ----------
    logger : logging.Logger
        The shared pipeline logger injected at construction time.
    """

    def __init__(self, name: str, logger: logging.Logger) -> None:
        super().__init__(name=name)
        self.logger = logger

    def debug(self, msg, *args, **kwargs):
        """Emit a DEBUG-level log message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Emit an INFO-level log message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Emit a WARNING-level log message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Emit an ERROR-level log message."""
        self.logger.error(msg, *args, **kwargs)


# ---------------------------------------------------------------------------
# Pipeline thread 1 — Reader
# ---------------------------------------------------------------------------

class Reader(LoggingThread):
    """Decodes video frames and pushes them onto the reading queue.

    Supports three input modes:
    * **Video file** — any format supported by OpenCV (MP4, AVI, MKV, …).
    * **Standard webcam / USB camera** — via an integer device index.
    * **Raspberry Pi camera** — via the ``picamera2`` library when
      ``raspberrypi_camera=True``.

    For video files, frames can be decoded far faster than the downstream
    MotionDetector can process them, so the Reader monitors the queue size
    and sleeps when the queue exceeds ``flush_proportion`` of its capacity.
    This "smart sleeping" prevents the reading queue from filling up and
    causing memory pressure, at the cost of a small amount of additional
    latency before each burst of frames is processed.

    For live camera feeds, sleeping is dangerous because it drops real-time
    footage.  The flush threshold should therefore be set high (e.g. 0.9)
    and the processing pipeline must be fast enough to keep up with the
    camera's frame rate.

    Termination is signalled to downstream threads by placing a ``None``
    sentinel on the reading queue and setting ``stop_signal``.

    Attributes
    ----------
    reading_queue : Queue
        Bounded queue shared with the MotionDetector.
    flush_thresh : int
        Absolute queue size (frames) at which the Reader begins sleeping,
        derived from ``flush_proportion * reading_queue.maxsize``.
    cam_feed : bool
        True when reading from a live camera (integer source); False for
        file input.
    picam_feed : bool
        True when using the Raspberry Pi ``picamera2`` back-end.
    vc : cv2.VideoCapture or Picamera2
        The underlying capture object.
    frame_count : int
        Running count of frames read, used for periodic FPS logging.
    """

    def __init__(
        self,
        reading_queue: Queue,
        video_source: Union[str, int],
        camera_resolution: Tuple[int, int],
        camera_fps: int,
        raspberrypi_camera: bool,
        record_duration: int,
        stop_signal: Event,
        sleep_seconds: int,
        flush_proportion: float,
        logger: logging.Logger,
    ) -> None:
        """Initialise the Reader and open the video capture.

        Parameters
        ----------
        reading_queue : Queue
            Bounded queue onto which decoded frames are pushed.  Must have a
            finite ``maxsize`` so that ``flush_proportion`` can be applied.
        video_source : str or int
            File path (str) or camera device index (int).  0 selects the
            default webcam on most systems.
        camera_resolution : Tuple[int, int]
            ``(width, height)`` used when initialising a live camera or
            Raspberry Pi camera.  Ignored for file input.
        camera_fps : int
            Target frame rate for live camera capture.  Ignored for file
            input.
        raspberrypi_camera : bool
            When True and ``video_source`` is an integer, the Raspberry Pi
            ``picamera2`` library is used instead of OpenCV VideoCapture.
        record_duration : int
            For live camera sources only: stop reading after this many
            seconds.
        stop_signal : Event
            Shared threading Event.  The Reader checks this on every frame
            and exits its loop when it is set.  The Reader also *sets* this
            signal when it finishes, to notify downstream threads.
        sleep_seconds : int
            Initial sleep duration when the queue fills up.  Updated
            dynamically by ``smart_sleep`` (currently unused).
        flush_proportion : float
            Queue occupancy fraction (0–1) that triggers a sleep.  Converts
            to ``flush_thresh = int(flush_proportion * reading_queue.maxsize)``.
        logger : logging.Logger
            Shared pipeline logger.
        """
        super().__init__(name="ReaderThread", logger=logger)

        self.reading_queue = reading_queue
        self.video_source = video_source
        self.record_duration = record_duration
        self.camera_resolution = camera_resolution
        self.picam_feed = raspberrypi_camera
        self.camera_fps = camera_fps
        self.stop_signal = stop_signal
        self.sleep_seconds = sleep_seconds
        self.frame_count = 0

        # Convert the proportional threshold to an absolute frame count.
        self.flush_thresh = int(flush_proportion * reading_queue.maxsize)

        # Open the capture back-end immediately so that get_fps() and
        # get_frame_size() are available to the Writer before run() is called.
        try:
            self.vc, self.cam_feed = self.get_video_capture(source=self.video_source)
            if self.cam_feed is True and self.picam_feed is True:
                # Replace the OpenCV capture with a Raspberry Pi camera instance.
                # FrameDurationLimits is specified in microseconds.
                from picamera2 import Picamera2

                pifps = round((1 / self.camera_fps) * 1000000)
                cam_setup = {"size": self.camera_resolution, "format": "RGB888"}
                self.vc = Picamera2()
                video_config_cam = self.vc.create_video_configuration(
                    main=cam_setup,
                    controls={"FrameDurationLimits": (pifps, pifps)},
                )
                self.vc.configure(video_config_cam)
                self.info(f"Camera resolution: {self.camera_resolution}, FPS: {self.camera_fps}")
            else:
                # picamera2 is only used when reading from a camera device,
                # not from a file.
                self.picam_feed = False

        except ValueError:
            # Signal failure to downstream threads immediately so they can
            # exit rather than blocking on an empty queue indefinitely.
            self.stop_signal.set()
            self.reading_queue.put(None)
            self.error(f"Could not make VideoCapture from source '{video_source}'")

        self.info(
            f"Will sleep {self.sleep_seconds} seconds if reading queue fills up with "
            f"{self.flush_thresh} frames. This *should not happen* if you're using a live "
            f"webcam, else the frames are being processed too slowly!"
        )

    def run(self) -> None:
        """Read frames in a loop and push each one onto the reading queue.

        For live camera feeds, ``check_recording_complete`` enforces the
        ``record_duration`` time limit.  For video files the loop ends
        naturally when ``vc.read()`` returns ``None`` (end of file).

        When the queue is close to full (qsize >= flush_thresh) a warning is
        logged.  The actual sleep call (smart_sleep) is currently disabled;
        the log messages remain for diagnostic purposes.

        After the loop exits the ``None`` sentinel is pushed to notify the
        MotionDetector, the capture is released, and ``stop_signal`` is set.
        """
        self.start_time = time.monotonic()

        if self.picam_feed is True:
            self.vc.start()

        while True:
            time_now = time.monotonic()

            if self.stop_signal.is_set():
                self.info("Received stop signal")
                break

            if self.picam_feed is True:
                frame = self.vc.capture_array()
            else:
                _, frame = self.vc.read()

            # End of file or time limit reached — exit the read loop.
            if frame is None or self.check_recording_complete(time_now):
                break

            if self.cam_feed is True:
                self.frame_count += 1
                if self.frame_count % 150 == 0:
                    self.info(
                        f"Read {self.frame_count} frames so far. "
                        f"FPS: {self.calculate_fps(self.start_time, time_now)}"
                    )

            # For live feeds a full queue means dropped footage.  For file
            # input it just means the MotionDetector is the bottleneck, which
            # is expected and harmless.
            if self.reading_queue.qsize() >= self.flush_thresh:
                self.debug(
                    f"Queue filled up to threshold. Sleeping {self.sleep_seconds} seconds "
                    f"to make sure queue can be drained..."
                )
                self.debug(
                    f"Finished sleeping with {self.reading_queue.qsize()} frames still in buffer!"
                )

            self.reading_queue.put(frame)

        # Push the sentinel value so the MotionDetector knows to stop.
        self.info("Adding None to end of reading queue")
        self.reading_queue.put(None)

        # Release the capture resource.
        if self.picam_feed is True:
            self.vc.stop()
            self.vc.close()
        else:
            self.vc.release()

        # Notify any other threads watching stop_signal (e.g. main's poll loop).
        self.stop_signal.set()

    def get_fps(self) -> int:
        """Return the frame rate of the current video source.

        For Raspberry Pi camera, this returns the configured FPS.  For all
        other sources it queries the OpenCV VideoCapture property, which
        reflects the actual frame rate embedded in the file or reported by
        the camera driver.
        """
        if self.picam_feed is True:
            return self.camera_fps
        else:
            return int(self.vc.get(cv2.CAP_PROP_FPS))

    def check_recording_complete(self, time_now: float) -> bool:
        """Return True when the live-camera recording time limit is exceeded.

        For file sources (``cam_feed=False``) this always returns False
        because the end of the file is detected by ``vc.read()`` returning
        None.

        Parameters
        ----------
        time_now : float
            Current monotonic timestamp (seconds), as returned by
            ``time.monotonic()``.
        """
        if self.cam_feed and (time_now - self.start_time >= self.record_duration):
            return True
        else:
            return False

    def calculate_fps(self, start_time: float, end_time: float) -> float:
        """Compute the observed read rate in frames per second.

        Parameters
        ----------
        start_time : float
            Monotonic timestamp when reading started.
        end_time : float
            Current monotonic timestamp.
        """
        return round(self.frame_count / (end_time - start_time), 2)

    def get_frame_size(self) -> Tuple[int, int]:
        """Return the ``(width, height)`` of frames produced by this Reader.

        Called by ``Writer.from_reader()`` before the threads are started so
        the VideoWriter can be configured with the correct dimensions.
        """
        if self.picam_feed is True:
            return self.camera_resolution
        else:
            width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)

    @staticmethod
    def get_video_capture(source: Union[str, int]) -> Tuple[cv2.VideoCapture, bool]:
        """Open an OpenCV VideoCapture from a file path or camera index.

        Parameters
        ----------
        source : str or int
            A file path string, or a non-negative integer camera device index.

        Returns
        -------
        (cv2.VideoCapture, bool)
            The VideoCapture object and a flag that is True when ``source``
            is a live camera (integer) and False for file input.

        Raises
        ------
        ValueError
            If ``source`` is neither a string nor an integer.
        """
        if type(source) is str:
            return cv2.VideoCapture(filename=source), False
        elif type(source) is int:
            return cv2.VideoCapture(index=source), True
        else:
            raise ValueError(
                "`source` must be a filepath to a video, or an integer index for the camera"
            )

    def smart_sleep(self, sleep_seconds: int, queue: Queue) -> int:
        """Adaptively adjust the sleep duration based on queue occupancy.

        After sleeping for ``sleep_seconds``, the queue size is re-evaluated.
        If the queue drained too quickly (size below the target range), the
        sleep duration is shortened for next time.  If the queue is still
        very full, the duration is extended.  This aims to keep the queue
        in the 5–15 % occupancy range after each sleep cycle.

        Note: this method is currently not called — the sleep in ``run()``
        was disabled while evaluating performance.  The method is retained
        here for future use.

        Parameters
        ----------
        sleep_seconds : int
            Seconds to sleep.
        queue : Queue
            The reading queue whose occupancy is being managed.

        Returns
        -------
        int
            Adjusted sleep duration for the next call.  Never negative.
        """
        qsize_before = queue.qsize()
        time.sleep(sleep_seconds)
        self.debug("WOKE UP!")
        qsize_after = queue.qsize()

        # Target occupancy band: 5–15 % of the queue's maximum capacity.
        lower_qsize = int(0.05 * queue.maxsize)
        upper_qsize = int(0.15 * queue.maxsize)

        # Queue drained to exactly the target band — current sleep is ideal.
        if qsize_after >= lower_qsize and qsize_after <= upper_qsize:
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; within range of "
                f"{lower_qsize}-{upper_qsize}; sleep of {sleep_seconds} sec was ideal!"
            )
            return sleep_seconds

        # Queue is nearly empty: we slept far too long — reduce aggressively.
        if qsize_after <= 2:
            new_sleep_seconds = sleep_seconds * 0.90
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *below* range of "
                f"{lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds

        # Queue fell below the lower band — reduce sleep slightly.
        if qsize_after < lower_qsize:
            new_sleep_seconds = sleep_seconds * 0.95
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *below* range of "
                f"{lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds

        # Queue is above the upper band — extend sleep to allow more draining.
        if qsize_after > upper_qsize:
            new_sleep_seconds = sleep_seconds * 1.05
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *above* range of "
                f"{lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds


# ---------------------------------------------------------------------------
# Pipeline thread 3 — Writer
# ---------------------------------------------------------------------------

class Writer(LoggingThread):
    """Encodes motion frames to disk and writes a CSV sidecar.

    Receives ``(frame, original_frame_number, first_in_seq, full_frame_recorded)``
    tuples from the writing queue, applies optional timestamp overlays and
    JPEG frame saving, then encodes each frame to the output video.

    Two encoding back-ends are supported:

    * **OpenCV** (default) — uses ``cv2.VideoWriter`` with a FOURCC codec.
      Works for DIVX, X264, FFV1, MJPEG, and most other common codecs.
    * **FFmpeg** (fallback) — spawns an ``ffmpeg`` subprocess that reads raw
      BGR frames from stdin.  Used automatically for HEVC/H.265, and as a
      fallback if OpenCV fails to open the VideoWriter (e.g. the build of
      OpenCV lacks a particular codec).

    After the last frame the Writer releases the video file and writes a CSV
    sidecar (``<filename>_video_info.csv``) whose columns are:

    * ``frame_number`` — sequential index in the *output* video (1-based).
    * ``original_frame_number`` — the corresponding frame index in the
      *input* video.
    * ``frame_with_full_frame`` — output frame number of the most recent
      full keyframe recorded at the start of this motion sequence, or empty
      if the motion sequence started with a masked frame.

    Attributes
    ----------
    writing_queue : Queue
        Shared queue between MotionDetector and Writer.
    filepath : str
        Absolute path to the output video file.
    frame_size : Tuple[int, int]
        ``(width, height)`` of output frames.
    fps : int
        Frame rate of the output video (inherited from the input source).
    backend : str
        Active encoding back-end, either ``"opencv"`` or ``"ffmpeg"``.
    frame_count : int
        Running count of frames written to the output video.
    """

    def __init__(
        self,
        writing_queue: Queue,
        filepath: str,
        frame_size: Tuple[int, int],
        fps: int,
        stop_signal: Event,
        logger: logging.Logger,
        output_filename: str,
        video_codec: str,
        embed_timestamps: bool,
        save_frames: bool,
        frames_to_save: int,
    ) -> None:
        """Initialise the Writer and select an encoding back-end.

        Parameters
        ----------
        writing_queue : Queue
            Queue from which motion frames are consumed.  A ``None`` sentinel
            signals that no more frames are coming.
        filepath : str
            Full path to the output video file, including extension.
        frame_size : Tuple[int, int]
            ``(width, height)`` of each frame, used to configure the encoder.
        fps : int
            Output video frame rate (should match the input source).
        stop_signal : Event
            Shared Event; the Writer exits on an empty queue when this is set.
        logger : logging.Logger
            Shared pipeline logger.
        output_filename : str
            Base filename (no extension) used for naming the CSV sidecar and
            JPEG frame files.
        video_codec : str
            Codec identifier string (e.g. ``"DIVX"``, ``"HEVC"``).
        embed_timestamps : bool
            Burn frame number and wall-clock time into each output frame.
        save_frames : bool
            Save individual motion frames as JPEG images.
        frames_to_save : int
            Maximum number of JPEG frames to save.  Ignored if
            ``save_frames`` is False.
        """
        super().__init__(name="WriterThread", logger=logger)

        self.writing_queue = writing_queue
        self.filepath = filepath
        self.frame_size = frame_size
        self.fps = fps
        self.stop_signal = stop_signal
        self.output_filename = output_filename
        self.embed_timestamps = embed_timestamps
        self.save_frames = save_frames
        self.frames_to_save = frames_to_save if save_frames else 0
        self.video_codec = (video_codec or "").strip()
        self.frame_count = 0

        # Choose between the OpenCV and FFmpeg encoding back-ends.
        self.backend: str = self._select_backend(self.video_codec)
        self.info(f"Selected writer backend: {self.backend}")

        # Pre-compute the FOURCC integer so OpenCV VideoWriter can be opened
        # immediately when run() starts.
        self._cv_fourcc: Optional[int] = None
        if self.backend == "opencv":
            codec_tag = self.video_codec if len(self.video_codec) in (0, 4) else "mp4v"
            if not codec_tag:
                codec_tag = "mp4v"
            self._cv_fourcc = cv2.VideoWriter_fourcc(*codec_tag.upper())
            self.info(f"OpenCV codec (FOURCC): {codec_tag.upper()}")

        # FFmpeg process handle and encoder name (used for logging only).
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_encoder_name: Optional[str] = None

    @classmethod
    def from_reader(
        cls,
        reader: Any,
        writing_queue: Queue,
        filepath: str,
        stop_signal: Event,
        logger: logging.Logger,
        output_filename: str,
        video_codec: str,
        embed_timestamps: bool,
        save_frames: bool,
        frames_to_save: int,
    ) -> "Writer":
        """Construct a Writer whose FPS and frame size are inherited from a Reader.

        This is the standard way to create a Writer in the pipeline.  The
        Reader is queried for its FPS and frame dimensions so the output
        video matches the input source exactly.

        Parameters
        ----------
        reader : Reader
            An initialised Reader (capture must already be open).
        writing_queue : Queue
            Queue from which motion frames are consumed.
        filepath : str
            Full output file path including extension.
        stop_signal : Event
            Shared stop Event.
        logger : logging.Logger
            Shared pipeline logger.
        output_filename : str
            Base filename for sidecar files.
        video_codec : str
            Codec identifier (e.g. ``"DIVX"``).
        embed_timestamps : bool
            Burn timestamps into frames if True.
        save_frames : bool
            Save JPEG snapshots if True.
        frames_to_save : int
            Maximum JPEG snapshots to save.
        """
        fps = reader.get_fps()
        frame_size = reader.get_frame_size()
        return Writer(
            writing_queue=writing_queue,
            filepath=filepath,
            frame_size=frame_size,
            fps=fps,
            stop_signal=stop_signal,
            logger=logger,
            output_filename=output_filename,
            video_codec=video_codec,
            embed_timestamps=embed_timestamps,
            save_frames=save_frames,
            frames_to_save=frames_to_save,
        )

    # ---------------- Backend selection ----------------

    def _select_backend(self, codec: str) -> str:
        """Return ``"ffmpeg"`` for HEVC and ``"opencv"`` for everything else.

        OpenCV's VideoWriter cannot encode H.265/HEVC in most builds, so
        those codecs are routed through an FFmpeg subprocess.  All other
        supported codecs (DIVX, X264, FFV1, MJPEG, …) are handled by OpenCV.
        If OpenCV subsequently fails to open the VideoWriter at runtime, the
        Writer falls back to FFmpeg automatically in ``run()``.
        """
        hevc_names = {"H265", "HEVC", "HVC1", "HEV1"}
        if (codec or "").upper() in hevc_names:
            return "ffmpeg"
        return "opencv"

    # ---------------- FFmpeg helpers ----------------

    def _map_codec_to_ffmpeg(self, ext: str) -> List[str]:
        """Map the user codec string to FFmpeg ``-c:v`` encoder arguments.

        Also sets ``self._ffmpeg_encoder_name`` for logging.  For H.265 in
        MP4 containers, a ``-tag:v hvc1`` argument is added so Apple devices
        can play the file.

        Parameters
        ----------
        ext : str
            Output file extension (e.g. ``".mp4"``), used to decide whether
            the Apple H.265 tag is needed.

        Returns
        -------
        List[str]
            FFmpeg argument fragments to be spliced into the command line,
            e.g. ``["-c:v", "libx265", "-tag:v", "hvc1"]``.
        """
        tag_args: List[str] = []
        v = self.video_codec.upper() if self.video_codec else ""

        if v in {"H264", "AVC1", "X264"}:
            enc = "libx264"
        elif v in {"H265", "HEVC", "HVC1", "HEV1"}:
            enc = "libx265"
            if ext in {".mp4", ".mov", ".m4v"}:
                # Required for H.265 playback on Apple devices.
                tag_args = ["-tag:v", "hvc1"]
        elif v in {"MPEG4", "MP4V", "XVID", "DIVX"}:
            enc = "mpeg4"
        elif v in {"MJPG", "MJPEG"}:
            enc = "mjpeg"
        else:
            self.warning(f"Unknown codec '{self.video_codec}', defaulting to H.264 (libx264).")
            enc = "libx264"

        self._ffmpeg_encoder_name = enc
        return ["-c:v", enc, *tag_args]

    def _build_ffmpeg_cmd(self) -> List[str]:
        """Construct the full FFmpeg command for a raw-video pipe.

        FFmpeg is configured to read raw BGR frames from stdin (``-i -``) at
        the correct resolution and frame rate, encode them with the chosen
        codec, and write the result to ``self.filepath``.

        The ``-movflags +faststart`` flag is added for MP4 output so the
        moov atom is placed at the start of the file, enabling streaming
        playback before the file is fully downloaded.

        Returns
        -------
        List[str]
            The complete command as a list of strings suitable for
            ``subprocess.Popen``.
        """
        width, height = self.frame_size
        ext = Path(self.filepath).suffix.lower()

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",      # Suppress FFmpeg's verbose startup banner.
            "-y",                       # Overwrite output file without prompting.
            "-f", "rawvideo",           # Input format: raw uncompressed video.
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",        # OpenCV uses BGR byte order.
            "-s", f"{width}x{height}",  # Frame dimensions.
            "-r", str(self.fps),        # Input frame rate (must match source).
            "-i", "-",                  # Read from stdin.
            *self._map_codec_to_ffmpeg(ext),
            "-pix_fmt", "yuv420p",      # Most compatible output pixel format.
        ]
        if ext in {".mp4", ".m4v"}:
            # Place the MP4 moov atom at the front for progressive playback.
            cmd += ["-movflags", "+faststart"]
        cmd += [self.filepath]
        return cmd

    def _start_ffmpeg(self) -> None:
        """Launch the FFmpeg subprocess with stdin connected as a raw-frame pipe.

        Does nothing if the subprocess is already running.  The subprocess's
        stdout is discarded and stderr is captured so that error messages can
        be included in any ``RuntimeError`` raised during frame writing.
        """
        if self._ffmpeg_proc is not None:
            return
        cmd = self._build_ffmpeg_cmd()
        self.info(f"FFmpeg encoder: {self._ffmpeg_encoder_name}")
        self.debug("Starting FFmpeg: " + " ".join(cmd))
        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=10_485_760,  # 10 MB pipe buffer to reduce write-blocking.
        )

    def _write_frame_ffmpeg(self, frame: np.ndarray) -> None:
        """Write a single BGR frame to the FFmpeg stdin pipe.

        Resizes the frame if its dimensions do not match ``self.frame_size``
        (e.g. if the MotionDetector returned a differently-shaped array).

        Parameters
        ----------
        frame : np.ndarray
            BGR image array with dtype uint8.

        Raises
        ------
        RuntimeError
            If the FFmpeg process has not been started or if the pipe breaks.
            The FFmpeg stderr output is included in the error message to aid
            debugging.
        """
        if self._ffmpeg_proc is None or self._ffmpeg_proc.stdin is None:
            raise RuntimeError("FFmpeg process not started or stdin unavailable.")
        h, w, c = frame.shape
        exp_w, exp_h = self.frame_size
        if (w, h) != (exp_w, exp_h) or c != 3:
            frame = cv2.resize(frame, (exp_w, exp_h), interpolation=cv2.INTER_LINEAR)
        try:
            self._ffmpeg_proc.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            # Attempt to read FFmpeg's stderr for a human-readable error message.
            err = b""
            if self._ffmpeg_proc and self._ffmpeg_proc.stderr:
                try:
                    err = self._ffmpeg_proc.stderr.read()
                except Exception:
                    pass
            raise RuntimeError(
                f"FFmpeg pipe broke while writing frame: {e}\n{err.decode(errors='ignore')}"
            )

    def _close_ffmpeg(self) -> None:
        """Close the FFmpeg pipe and wait for the process to finish encoding.

        Closing stdin signals FFmpeg that no more frames are coming, which
        causes it to flush its encoder buffers and write the final output
        file.  A 30-second timeout is enforced; if FFmpeg does not exit
        within that time a warning is logged and the reference is cleared.
        """
        if self._ffmpeg_proc is None:
            return
        try:
            if self._ffmpeg_proc.stdin:
                self._ffmpeg_proc.stdin.close()
            if self._ffmpeg_proc.stderr:
                try:
                    # Drain stderr to prevent the subprocess from blocking on
                    # a full pipe buffer.
                    _ = self._ffmpeg_proc.stderr.read()
                except Exception:
                    pass
            rc = self._ffmpeg_proc.wait(timeout=30)
            if rc != 0:
                self.warning(
                    f"FFmpeg exited with code {rc}. Output may be incomplete: {self.filepath}"
                )
        except Exception as e:
            self.warning(f"Error closing FFmpeg: {e}")
        finally:
            self._ffmpeg_proc = None

    # ---------------- Main loop ----------------

    def run(self) -> None:
        """Consume frames from the writing queue and encode them to disk.

        The loop blocks on ``writing_queue.get(timeout=0.5)`` so it wakes up
        promptly when a new frame arrives, rather than spin-waiting on a
        size threshold.  This ensures that sparse motion frames (e.g. one
        per second) are written to disk immediately rather than accumulating
        until a batch threshold is reached.

        Queue protocol
        --------------
        Each item in the writing queue is a list::

            [frame, nframe, first_in_seq, ff_recorded]

        * ``frame`` — BGR ndarray, already masked by MotionDetector.
        * ``nframe`` — original frame index in the input video.
        * ``first_in_seq`` — True if this is the first motion frame after a
          period of no motion.
        * ``ff_recorded`` — True if this frame is an unmasked keyframe
          (written every ``full_frame_capture_interval`` frames).

        A ``None`` item is the sentinel that signals the end of the stream.

        After the last frame, the video file is finalised and a CSV sidecar
        is written that maps each output frame back to its original index.
        """
        # Open the chosen encoding back-end.
        vw = None
        if self.backend == "opencv":
            vw = cv2.VideoWriter(
                filename=self.filepath,
                fourcc=self._cv_fourcc if self._cv_fourcc is not None else 0,
                fps=self.fps,
                frameSize=self.frame_size,
            )
            if not vw.isOpened():
                # OpenCV can fail silently (e.g. codec not in build); fall
                # back to FFmpeg so we still produce output.
                self.warning("OpenCV VideoWriter failed to open; falling back to FFmpeg.")
                self.backend = "ffmpeg"
                self._start_ffmpeg()
                self.info(f"Switched writer backend: {self.backend}")
            else:
                self.info("Using OpenCV VideoWriter.")
        else:
            self._start_ffmpeg()
            self.info("Using FFmpeg pipe.")

        saved_frames = 0
        # Accumulates rows for the CSV sidecar: each entry records the
        # output frame number, original frame number, and (if applicable)
        # the output frame number of the most recent keyframe.
        frame_info: List[List[Optional[int]]] = []

        while True:
            try:
                frame_combo = self.writing_queue.get(timeout=0.5)
            except Empty:
                # No frame arrived within the timeout window.  If the stop
                # signal is set and the queue is empty we can safely exit.
                if self.stop_signal.is_set():
                    self.warning("Stop signal received with empty writing queue. Finalising output.")
                    break
                continue

            if frame_combo is None:
                # Sentinel: MotionDetector has finished processing all frames.
                break

            frame, nframe, first_in_seq, ff_recorded = frame_combo
            self.frame_count += 1

            if self.embed_timestamps:
                # Overlay compressed frame number, original frame number and
                # wall-clock time derived from the original frame index.
                raw_time = nframe / float(self.fps)
                minutes, seconds = divmod(raw_time, 60)
                cv2.putText(
                    frame,
                    text=(
                        f"Frame: {self.frame_count}  |  "
                        f"Raw Frame: {nframe}  |  "
                        f"Time: {int(minutes):02d}:{int(seconds):02d}"
                    ),
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=2,
                )

            if self.save_frames and saved_frames < self.frames_to_save:
                image_filepath = (
                    Path(self.filepath).parent
                    / f"{self.output_filename}_frame_{self.frame_count}.jpg"
                )
                cv2.imwrite(str(image_filepath), frame)
                saved_frames += 1

            if self.backend == "opencv" and vw is not None:
                vw.write(frame)
            else:
                self._write_frame_ffmpeg(frame)

            if first_in_seq:
                # Record the keyframe number for this motion sequence start.
                # nff_number is None if this first frame was a masked frame
                # rather than a full-resolution keyframe.
                nff_number: Optional[int] = self.frame_count if ff_recorded else None
                frame_info.append([self.frame_count, nframe, nff_number])

            if self.frame_count % 50 == 0:
                self.info(f"Written {self.frame_count} frames so far")

        # Flush encoder buffers and close the file handle.
        if self.backend == "opencv" and vw is not None:
            vw.release()
        else:
            self._close_ffmpeg()

        # Write the CSV sidecar so downstream analysis can map compressed
        # frame indices back to original frame numbers.
        csv_filepath = Path(self.filepath).parent / f"{self.output_filename}_video_info.csv"
        with open(csv_filepath, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["frame_number", "original_frame_number", "frame_with_full_frame"])
            for row in frame_info:
                csv_writer.writerow(row)


# Maps codec names to their preferred output container extension.
# FFV1 is a lossless codec typically stored in AVI containers because
# standard MP4 does not support it.  HEVC variants all use MP4.
_CODEC_EXT: dict = {
    "FFV1": ".avi",
    "HEVC": ".mp4",
    "H265": ".mp4",
    "HVC1": ".mp4",
    "HEV1": ".mp4",
}


# ---------------------------------------------------------------------------
# Pipeline thread 2 — MotionDetector
# ---------------------------------------------------------------------------

class MotionDetector(LoggingThread):
    """Filters video frames to retain only those containing motion.

    Sits between the Reader and the Writer in the pipeline.  For each
    incoming frame it computes a binary motion mask by differencing
    consecutive frames, and only forwards frames where motion is detected
    to the writing queue.

    Algorithm overview
    ------------------
    1. **Downscale** — the frame is scaled down by ``downscale_factor`` and
       converted to greyscale.  Working at reduced resolution dramatically
       cuts the cost of the subsequent morphological operations.

    2. **Frame difference** — the absolute per-pixel difference between the
       current downscaled frame and the previous one is computed.  Large
       differences indicate movement.

    3. **Dilation** — the difference image is morphologically dilated with a
       square kernel.  This expands bright (high-difference) regions,
       bridging small gaps and making the mask more robust to sub-pixel
       movement.

    4. **Threshold** — pixels are binarised: a value of 255 if the difference
       exceeds ``movement_threshold``, otherwise 0.

    5. **Second dilation + median blur** — the binary mask is dilated again
       and smoothed with a 9×9 median filter to remove isolated noise pixels
       and fill holes within moving objects.

    6. **Motion decision** — ``cv2.countNonZero`` tests whether any pixel in
       the final mask is non-zero.  If yes, motion is present.

    7. **Frame construction** — motion frames are built at the *original*
       (un-downscaled) resolution using ``cv2.bitwise_and``, zeroing out
       all background pixels outside the mask.  If ``background_transparency``
       > 0, the background is partially blended in rather than set to pure
       black (see below).

    8. **Keyframes** — every ``full_frame_capture_interval`` input frames, or
       at the very first frame of a motion sequence, a full unmasked copy of
       the original frame is saved.  Keyframes provide spatial context that
       is otherwise lost when masking.

    9. **Post-motion buffer** — after motion stops, up to
       ``post_motion_record_frames`` additional frames are written using the
       *last known* motion mask.  This captures the tail of an event (e.g.
       an animal walking out of frame).

    Background transparency
    -----------------------
    When ``background_transparency`` > 0, non-motion background pixels are
    not set to black but instead receive a darkened copy of the original
    frame.  The blend is::

        bg = frame × alpha              (alpha = background_transparency)
        motion_frame = bitwise_and(frame, mask)
        output = absdiff(motion_frame, bg) + motion_frame

    In *motion* regions (where bitwise_and preserves the original pixel):
        output = |frame − frame×alpha| + frame = frame×(1−alpha) + frame
                = frame × (2 − alpha)

    In *background* regions (where bitwise_and outputs 0):
        output = |0 − frame×alpha| + 0 = frame × alpha

    Net result: motion regions are slightly brightened (or unchanged when
    alpha = 1), and background regions are shown at ``alpha`` brightness.
    At the default ``background_transparency = 1.0``, the full original
    frame is reproduced (effectively no compression).

    Attributes
    ----------
    input_queue : Queue
        Reading queue shared with the Reader.
    writing_queue : Queue
        Writing queue shared with the Writer.
    nframe : int
        Count of input frames processed so far.
    prev_frame : np.ndarray or None
        Downscaled greyscale frame from the previous iteration.
    prev_mask : np.ndarray or None
        Binary motion mask from the previous iteration, stored at downscale
        resolution for consistent sizing.
    buffer_analysis : bool
        True when post-motion buffer recording is enabled.
    buffer_count : int
        Number of buffer frames already written after the last motion event.
    record_full_frame : bool
        Flag indicating that the next motion frame should be a full keyframe.
    transparent_background : bool
        True when background transparency blending is active.
    dilation_kernel : np.ndarray
        Square ones-array used as the morphological dilation kernel,
        pre-scaled to the downscale resolution.
    """

    def __init__(
        self,
        input_queue: Queue,
        writing_queue: Queue,
        downscale_factor: int,
        dilate_kernel_size: int,
        movement_threshold: int,
        post_motion_record_frames: float,
        full_frame_capture_interval: int,
        background_transparency: float,
        stop_signal: Event,
        logger: logging.Logger,
    ) -> None:
        """Initialise the MotionDetector with detection parameters.

        Parameters
        ----------
        input_queue : Queue
            Reading queue populated by the Reader.
        writing_queue : Queue
            Writing queue consumed by the Writer.
        downscale_factor : int
            Frames are resized by 1/downscale_factor before motion detection.
            The dilation kernel is also scaled accordingly.
        dilate_kernel_size : int
            Size of the dilation kernel in *original-resolution* pixels.
            Scaled to downscale resolution internally.
        movement_threshold : int
            Per-pixel intensity difference (0–255) required to flag a pixel
            as moving.
        post_motion_record_frames : float
            Extra frames to emit after motion stops.  Set to 0 to disable.
        full_frame_capture_interval : int
            Interval (in input frames) between unmasked keyframe captures.
        background_transparency : float
            Background blend level for non-keyframe motion frames (0–1).
        stop_signal : Event
            Shared threading Event checked on each frame.
        logger : logging.Logger
            Shared pipeline logger.
        """
        super().__init__(name="MotionThread", logger=logger)

        self.input_queue = input_queue
        self.writing_queue = writing_queue
        self.stop_signal = stop_signal
        self.nframe = 0
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_mask: Optional[np.ndarray] = None
        # Zero array used for the background transparency blend.  Allocated
        # lazily on the first frame because frame shape is not known yet.
        self._zero_frame: Optional[np.ndarray] = None

        # Motion detection parameters
        self.downscale_factor = downscale_factor
        # Scale factors for cv2.resize (reciprocals of downscale_factor).
        self.fx = self.fy = 1 / downscale_factor
        # Scale the kernel to the downscale resolution.  max(1, ...) prevents
        # a zero-size kernel when dilate_kernel_size < downscale_factor.
        downscaled_kernel_size = max(1, int(dilate_kernel_size / downscale_factor))
        self.dilation_kernel = np.ones((downscaled_kernel_size, downscaled_kernel_size))
        self.movement_threshold = movement_threshold
        self.post_motion_record_frames = post_motion_record_frames
        self.full_frame_capture_interval = full_frame_capture_interval
        self.background_transparency = background_transparency

        self.transparent_background: bool = background_transparency > 0
        # Buffer analysis is only meaningful when post_motion_record_frames > 0.
        self.buffer_analysis: bool = post_motion_record_frames > 0
        self.buffer_count: int = 0
        # Initialise to True so the very first motion frame is always a keyframe.
        self.record_full_frame: bool = True

        if downscale_factor != 1:
            self.info(
                f"Dilation kernel downscaled by {downscale_factor}x "
                f"from {dilate_kernel_size} to {self.dilation_kernel.shape[0]}"
            )

    def run(self) -> None:
        """Pull frames from the input queue and call detect_motion on each.

        Blocks on ``input_queue.get(timeout=10)`` so it does not spin-wait.
        If a 10-second timeout fires while the stop signal is not set, an
        error is logged (this indicates the Reader has stalled unexpectedly).

        After the ``None`` sentinel is received from the Reader, a matching
        ``None`` is pushed to the writing queue to notify the Writer.
        """
        while True:
            try:
                frame = self.input_queue.get(timeout=10)
            except Empty:
                if self.stop_signal.is_set():
                    self.error(
                        "Waited too long to get frame from input queue? This shouldn't happen!"
                    )
                continue
            if frame is None:
                break

            self.detect_motion(frame=frame)

        # Propagate the end-of-stream sentinel to the Writer.
        self.writing_queue.put(None)

    def detect_motion(self, frame: np.ndarray) -> None:
        """Analyse a single frame for motion and queue it if motion is detected.

        This is the core per-frame method.  See the class docstring for a
        full description of the algorithm.  Frames are only added to
        ``writing_queue`` when they contain motion (or are buffer frames
        immediately following motion).

        Each queued item is a list::

            [motion_frame, nframe, first_frame_in_seq, full_frame_recorded]

        Parameters
        ----------
        frame : np.ndarray
            Full-resolution BGR frame from the Reader.
        """
        self.nframe += 1
        first_frame_in_seq = False   # True only for the 1st frame of a motion sequence.
        full_frame_recorded = False  # True only if this frame is an unmasked keyframe.

        # Schedule a full keyframe at the configured interval and on frame 1.
        if (self.nframe % self.full_frame_capture_interval == 0) or (self.nframe == 1):
            self.record_full_frame = True

        # --- Step 1: build a downscaled greyscale frame for motion detection ---
        # Downscaling dramatically reduces the cost of the morphological
        # operations that follow.  All detection happens at reduced resolution;
        # the full-resolution frame is only used when constructing the output.
        if self.downscale_factor == 1:
            small_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            small_gray = cv2.cvtColor(
                cv2.resize(frame, dsize=None, fx=self.fx, fy=self.fy),
                cv2.COLOR_BGR2GRAY,
            )

        # Initialise prev_frame on the very first call (no diff possible yet).
        if self.prev_frame is None:
            self.prev_frame = small_gray

        # --- Steps 2–5: compute the cleaned-up binary motion mask ---
        # Absolute difference between current and previous greyscale frames.
        diff = cv2.absdiff(small_gray, self.prev_frame)
        # First dilation: expand difference regions to bridge nearby blobs.
        dilated = cv2.dilate(diff, kernel=self.dilation_kernel)
        # Threshold to binary: pixels with large enough difference become 255.
        _, thresh = cv2.threshold(dilated, self.movement_threshold, 255, cv2.THRESH_BINARY)
        # Second dilation + median blur: fill holes and remove isolated noise.
        mask_small = cv2.medianBlur(cv2.dilate(thresh, self.dilation_kernel), 9)

        # Initialise prev_mask on the very first call.
        if self.prev_mask is None:
            self.prev_mask = mask_small.copy()

        # --- Step 6: motion decision ---
        # Cache countNonZero results as booleans to avoid scanning the same
        # array multiple times within this method.
        has_motion: bool = cv2.countNonZero(mask_small) > 0
        had_motion: bool = cv2.countNonZero(self.prev_mask) > 0

        if has_motion:
            # --- Step 7: upscale mask and construct the output frame ---
            # The mask was computed at downscale resolution; it must be
            # upscaled to match the original frame before bitwise_and.
            if self.downscale_factor != 1:
                h, w = frame.shape[:2]
                mask_full = cv2.resize(mask_small, dsize=(w, h))
            else:
                mask_full = mask_small

            if self.transparent_background:
                # Build the darkened background blend target.
                # transparent_frame = frame × background_transparency.
                # Lazy-allocate the zeros array on the first motion frame.
                if self._zero_frame is None:
                    self._zero_frame = np.zeros_like(frame)
                transparent_frame = cv2.addWeighted(
                    frame,
                    self.background_transparency,
                    self._zero_frame,
                    1.0 - self.background_transparency,
                    0,
                )

            if not had_motion:
                # --- Step 8: first frame of a new motion sequence ---
                first_frame_in_seq = True
                if self.record_full_frame:
                    # Emit an unmasked keyframe for spatial context.
                    motion_frame = frame.copy()
                    self.record_full_frame = False
                    full_frame_recorded = True
                else:
                    # Emit a masked frame (background zeroed or darkened).
                    motion_frame = cv2.bitwise_and(frame, frame, mask=mask_full)
                    if self.transparent_background:
                        motion_frame = cv2.add(
                            cv2.absdiff(motion_frame, transparent_frame), motion_frame
                        )
            else:
                # Continuing an existing motion sequence — always masked.
                motion_frame = cv2.bitwise_and(frame, frame, mask=mask_full)
                if self.transparent_background:
                    motion_frame = cv2.add(
                        cv2.absdiff(motion_frame, transparent_frame), motion_frame
                    )

            # Store the mask at *downscale* resolution for consistent sizing
            # across all code paths (the buffer and no-motion paths also store
            # at downscale resolution).
            self.prev_mask = mask_small.copy()
            self.buffer_count = 0
            self.writing_queue.put([motion_frame, self.nframe, first_frame_in_seq, full_frame_recorded])

        elif self.buffer_analysis and had_motion:
            # --- Step 9: post-motion buffer ---
            # Motion has just stopped (had_motion=True, has_motion=False).
            # Continue emitting frames for `post_motion_record_frames` more
            # frames, applying the *last known* mask so the recently moving
            # region stays visible.
            if self.buffer_count < self.post_motion_record_frames:
                self.buffer_count += 1
                # prev_mask is at downscale resolution — upscale before use.
                if self.downscale_factor != 1:
                    h, w = frame.shape[:2]
                    mask_full = cv2.resize(self.prev_mask, dsize=(w, h))
                else:
                    mask_full = self.prev_mask
                motion_frame = cv2.bitwise_and(frame, frame, mask=mask_full)
                # Buffer frames are never the first in a sequence and never
                # full keyframes, so both metadata flags are False.
                self.writing_queue.put([motion_frame, self.nframe, False, False])
            else:
                # Buffer exhausted — transition back to idle state.
                self.prev_mask = mask_small.copy()
        else:
            # No motion and no active buffer — update state only.
            self.prev_mask = mask_small.copy()

        # Advance the previous-frame reference for the next iteration.
        self.prev_frame = small_gray.copy()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(config: Config) -> None:
    """Run one complete compression job for the given Config.

    Creates the output directory, configures logging, instantiates and starts
    the three pipeline threads (Reader, MotionDetector, Writer), then polls
    until they all finish.  Handles ``KeyboardInterrupt`` gracefully by
    setting the shared stop signal and waiting for threads to exit.

    Parameters
    ----------
    config : Config
        Fully populated Config object for this processing run.
    """
    start = time.time()

    # Reducing OpenCV's internal thread pool prevents it from competing with
    # the pipeline's own threads for CPU cores.
    cv2.setNumThreads(config.num_opencv_threads)

    # Derive the output filename stem from the input file (or a timestamp for
    # live camera sources).
    if type(config.video_source) is str:
        output_filename = os.path.splitext(os.path.basename(config.video_source))[0]
    else:
        output_filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine the output parent directory.  If the configured output
    # directory is valid, use it; otherwise fall back to a sub-folder next
    # to the input video file.
    if os.path.isdir(config.output_directory):
        output_parent_directory = Path(config.output_directory, "EcoMotionZip")
        log_message = f"Outputting to {output_parent_directory}"
    else:
        if os.path.isdir(str(config.video_source)):
            output_parent_directory = Path(str(config.video_source), "EcoMotionZip")
        else:
            output_parent_directory = Path(
                Path(str(config.video_source)).parent, "EcoMotionZip"
            )
        log_message = (
            f"Output directory not specified or unavailable. "
            f"Outputting to video source directory {output_parent_directory}"
        )

    os.makedirs(output_parent_directory, exist_ok=True)

    # The reading queue holds raw decoded frames; the writing queue holds
    # motion-masked frames ready for encoding.  Bounded sizes prevent
    # unbounded memory growth if any thread falls behind.
    reading_queue = Queue(maxsize=512)
    writing_queue = Queue(maxsize=256)
    stop_signal = Event()

    # Each processing run gets its own sub-directory inside the parent.
    output_directory = Path(output_parent_directory, output_filename)
    output_directory.mkdir(exist_ok=True)

    # Select the output container extension based on the codec.
    ext = _CODEC_EXT.get((config.video_codec or "").upper(), ".mp4")
    output_filepath = str(output_directory / f"{output_filename}{ext}")

    # Configure logging: INFO to console (with thread name prefix) and DEBUG
    # to a per-run log file in the output directory.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(threadName)-14s] %(msg)s"))
    file_handler = logging.FileHandler(filename=output_directory / f"{output_filename}_output.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(threadName)-14s] %(msg)s")
    )
    LOGGER.handlers.clear()
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)

    LOGGER.info(f"Starting processing at: {datetime.fromtimestamp(start)}")
    LOGGER.info(f"Running main() with Config: {config.__dict__}")
    LOGGER.info(log_message)

    # Construct all three threads.  The walrus operator binds names so that
    # `reader` can be passed to Writer.from_reader() without storing it
    # separately before building the tuple.
    threads = (
        reader := Reader(
            reading_queue=reading_queue,
            video_source=config.video_source,
            record_duration=config.record_duration,
            camera_resolution=config.camera_resolution,
            camera_fps=config.camera_fps,
            raspberrypi_camera=config.raspberrypi_camera,
            stop_signal=stop_signal,
            sleep_seconds=config.reader_sleep_seconds,
            flush_proportion=config.reader_flush_proportion,
            logger=LOGGER,
        ),
        motion_detector := MotionDetector(
            input_queue=reading_queue,
            writing_queue=writing_queue,
            downscale_factor=config.downscale_factor,
            dilate_kernel_size=config.dilate_kernel_size,
            movement_threshold=config.movement_threshold,
            post_motion_record_frames=config.post_motion_record_frames,
            full_frame_capture_interval=config.full_frame_capture_interval,
            background_transparency=config.background_transparency,
            stop_signal=stop_signal,
            logger=LOGGER,
        ),
        writer := Writer.from_reader(
            reader=reader,
            writing_queue=writing_queue,
            filepath=output_filepath,
            stop_signal=stop_signal,
            logger=LOGGER,
            output_filename=output_filename,
            video_codec=config.video_codec,
            save_frames=config.save_frames,
            frames_to_save=config.frames_to_save,
            embed_timestamps=config.embed_timestamps,
        ),
    )

    for thread in threads:
        LOGGER.info(f"Starting {thread.name}")
        thread.start()

    # Poll every 0.5 s until all threads have exited.  Keyboard interrupts
    # set the stop signal and break out of the loop; the join() calls below
    # then wait for the threads to finish gracefully.
    while True:
        try:
            time.sleep(0.5)
            if not any(thread.is_alive() for thread in threads):
                LOGGER.info("All child processes have finished! Exiting...")
                break
            for queue, queue_name in zip(
                [reading_queue, writing_queue],
                ["Reading", "Writing"],
            ):
                LOGGER.debug(f"{queue_name} queue size: {queue.qsize()}")
        except (KeyboardInterrupt, Exception):
            LOGGER.exception(
                "Received KeyboardInterrupt or Exception. Setting stop signal..."
            )
            LOGGER.warning("You may have to wait for all child processes to exit gracefully.")
            stop_signal.set()
            break

    for thread in threads:
        LOGGER.info(f"Joining {thread.name}")
        thread.join()

    end = time.time()
    LOGGER.info(f"Finished processing at: {datetime.fromtimestamp(end)}")
    LOGGER.info(f"Finished main() in {end - start:.2f} seconds.")

    if type(config.video_source) is str and config.delete_original_after_processing:
        os.remove(config.video_source)
        LOGGER.info(f"Deleted original video file: {config.video_source}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Extract the parameters that support list values for batch / grid-search
    # processing.  All other Config fields are held fixed across runs.
    downscale_factor = CONFIG.downscale_factor
    dilate_kernel_size = CONFIG.dilate_kernel_size
    movement_threshold = CONFIG.movement_threshold
    post_motion_record_frames = CONFIG.post_motion_record_frames
    full_frame_capture_interval = CONFIG.full_frame_capture_interval
    video_codec = CONFIG.video_codec
    video_source = CONFIG.video_source
    embed_timestamps = CONFIG.embed_timestamps

    # Resolve the video source into a flat list of file paths (or camera
    # indices).  A directory is expanded to all recognised video files within
    # it; a single file or integer is wrapped in a list for uniformity.
    if type(video_source) != int:
        video_source = Path(video_source)
        if video_source.is_dir():
            video_source = [
                str(v)
                for v in video_source.iterdir()
                if v.suffix in [".avi", ".mp4", ".h264", ".MTS"]
            ]
        elif type(video_source) is not list:
            video_source = [str(video_source)]
    else:
        # Repeat the camera index once per requested recording.
        video_source = [video_source] * CONFIG.number_of_videos

    # Wrap scalar detection parameters in lists so that itertools.product
    # can generate all combinations.  This allows a single config run to
    # sweep multiple parameter values (e.g. compare two threshold settings
    # across several videos) without any external scripting.
    if type(downscale_factor) is not list:
        downscale_factor = [downscale_factor]
    if type(dilate_kernel_size) is not list:
        dilate_kernel_size = [dilate_kernel_size]
    if type(movement_threshold) is not list:
        movement_threshold = [movement_threshold]
    if type(post_motion_record_frames) is not list:
        post_motion_record_frames = [post_motion_record_frames]
    if type(full_frame_capture_interval) is not list:
        full_frame_capture_interval = [full_frame_capture_interval]

    # Build the Cartesian product of all variable parameters.  For the
    # typical case (single video, single set of parameters) this produces
    # exactly one combination and main() is called once.
    parameter_combos = product(
        video_source,
        downscale_factor,
        dilate_kernel_size,
        movement_threshold,
        post_motion_record_frames,
        full_frame_capture_interval,
    )
    parameter_keys = [
        "video_source",
        "downscale_factor",
        "dilate_kernel_size",
        "movement_threshold",
        "post_motion_record_frames",
        "full_frame_capture_interval",
    ]

    for combo in parameter_combos:
        # Build a per-run Config by merging this combination's variable fields
        # with the fixed fields from the top-level CONFIG.
        this_config_dict = dict(zip(parameter_keys, combo))
        this_config_dict.update(
            {
                "output_directory": CONFIG.output_directory,
                "record_duration": CONFIG.record_duration,
                "number_of_videos": CONFIG.number_of_videos,
                "camera_resolution": CONFIG.camera_resolution,
                "camera_fps": CONFIG.camera_fps,
                "raspberrypi_camera": CONFIG.raspberrypi_camera,
                "delete_original_after_processing": CONFIG.delete_original_after_processing,
                "reader_sleep_seconds": CONFIG.reader_sleep_seconds,
                "reader_flush_proportion": CONFIG.reader_flush_proportion,
                "num_opencv_threads": CONFIG.num_opencv_threads,
                "video_codec": CONFIG.video_codec,
                "embed_timestamps": CONFIG.embed_timestamps,
                "background_transparency": CONFIG.background_transparency,
                "save_frames": CONFIG.save_frames,
                "frames_to_save": CONFIG.frames_to_save,
            }
        )
        this_config = Config(**this_config_dict)
        main(this_config)
