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
from typing import Tuple, Union, Optional
from itertools import product
import argparse
import subprocess
from typing import Tuple, List, Dict, Optional
import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class Config:
    """Configuration class to store parameters for video processing.

    Attributes:
        video_source (str): Path to the input directory or a single video file. Set value to 0 to use webcam or any other integer to use a different camera.
        output_directory (str): Path to the output directory.
        record_duration (int): Duration of the recording for a single video in seconds.
        number_of_videos (int): Number of videos to record.
        reader_sleep_seconds (float): Sleep duration for the reader thread in seconds.
        reader_flush_proportion (float): Proportion of the reading queue to be filled before the reader thread sleeps.
        downscale_factor (int): Downscale factor for input video.
        dilate_kernel_size (int): Kernel size for dilation.
        movement_threshold (int): Threshold for movement detection.
        post_motion_record_frames (float): Number of frames to persist for.
        full_frame_capture_interval (int): Number of frames to persist for.
        video_codec (str): Video codec to use for output video.
        num_opencv_threads (int): Number of threads to use for OpenCV.
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
        frames_to_save: int
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
    """
    Process input arguments and return them as a dictionary.

    Returns:
        dict: A dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process input and output directories.")
    parser.add_argument('--video_source', type=lambda x: int(x) if x.isdigit() else str(x), help='Path to the input directory or a single video file. Set value to 0 to use webcam or any other integer to use a different camera.')
    parser.add_argument('--output_directory', type=str, help='Path to the output directory')
    parser.add_argument('--record_duration', type=int, help='Duration of the recording for a single video in seconds.')
    parser.add_argument('--number_of_videos', type=int, help='Number of videos to record.')
    parser.add_argument('--camera_resolution', type=tuple, help='Resolution of the camera.')
    parser.add_argument('--camera_fps', type=int, help='FPS of the camera.')
    parser.add_argument('--delete_original_after_processing', type=bool, help='Delete original video after processing.')
    parser.add_argument('--downscale_factor', type=int, help='Downscale factor for input video.')
    parser.add_argument('--dilate_kernel_size', type=int, help='Kernel size for dilation.')
    parser.add_argument('--movement_threshold', type=int, help='Threshold for movement detection.')
    parser.add_argument('--post_motion_record_frames', type=float, help='Number of frames to persist for.')
    parser.add_argument('--full_frame_capture_interval', type=int, help='Number of frames to persist for.')
    parser.add_argument('--video_codec', type=str, choices=["DIVX", "X264"], help='Video codec to use for output video.')
    parser.add_argument('--num_opencv_threads', type=int, help='Number of threads to use for OpenCV.')
    parser.add_argument('--background_transparency', type=float, help='Background transparency.')
    parser.add_argument('--save_frames', type=bool, help='Save frames.')
    parser.add_argument('--frames_to_save', type=int, help='Number of frames to save.')

    args = parser.parse_args()

    return vars(args)

# Create Config object from JSON file
with open("config.json", "r") as f:
    __config_dict = json.load(f)

cmd_args = read_args()
__config_dict.update((k, v) for k, v in cmd_args.items() if v is not None)
CONFIG = Config(**__config_dict)




class LoggingThread(Thread):
    """A wrapper around `threading.Thread` with convenience methods for logging.
    
    This class extends the functionality of the `threading.Thread` class by providing
    additional convenience methods for logging. It serves as a base class for other
    threads that require logging capabilities.
    
    Attributes:
        name (str): The name of the thread.
        logger (logging.Logger): The logger object used for logging.
    """
    def __init__(self, name: str, logger: logging.Logger) -> None:
        super().__init__(name=name)

        self.logger = logger

    def debug(self, msg, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)


class Reader(LoggingThread):
    """A class to read video from either a file or camera, and push to a queue.
    
    Reads video from a file or camera and pushes each frame onto a queue. Note that
    this queue *must* be emptied before the program can be closed, meaning every frame
    in this queue needs to be handled by some other thread, either with actual processing
    or just popping those frames and doing nothing with them.

    This Reader includes "smart sleeping" to prevent the reading queue from filling up
    and blocking the reading thread. For video files, where frames can be read much more
    quickly than they can be processed, this thread will sleep for a while once the queue
    fills above a certain threshold (e.g. ~90%). This prevents wasting compute time by
    continuously trying to push frames onto a full queue.

    Attributes:
        reading_queue (Queue): The queue to push the video frames onto.
        video_source (Union[str, int]): The source of the video (file path or camera index).
        record_duration (int): The duration (in seconds) to record the video.
        stop_signal (Event): The event to signal when to stop reading the video.
        sleep_seconds (int): The number of seconds to sleep when the queue is full.
        flush_proportion (float): The proportion of frames to flush from the queue when it is full.
        logger (logging.Logger): The logger object used for logging.
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
        """Initialise Reader with given queue and video source.

        Parameters
        ----------
        reading_queue : Queue
            The queue to push video frames onto.
        video_source : Union[str, int]
            A string representing a video file's filepath, or a non-negative 
            integer index for an attached camera device. 0 is usually your 
            laptop/rapsberry pi's in-built webcam, but this will depend on the 
            hardware you're using.
        stop_signal : Event
            A threading Event that this Reader queries to know when to stop. 
            This is used for graceful termination of the multithreaded program.
        sleep_seconds : int
            *Initial* time to sleep. This sleep time will be dynamically updated
            according to smart sleep.
        flush_proportion : float
            When the queue fills above this proportion, trigger the sleeping to
            let other thread drain the queue. This prevents blocking when the
            queue becomes full. Recommended to be relatively high, e.g. 0.9.
        logger : logging.Logger
            Logger to use for logging key info, warnings, etc.
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

        # self.video_file_name = os.path.basename(video_source)
        
        self.flush_thresh = int(flush_proportion * reading_queue.maxsize)
        # Make video capture now so we can dynamically retrieve its FPS and frame size
        try:
            self.vc, self.cam_feed = self.get_video_capture(source=self.video_source)  
            if self.cam_feed is True and self.picam_feed is True:
                from picamera2 import Picamera2
                pifps = round((1/self.camera_fps)*1000000)
                cam_setup = {"size": self.camera_resolution, "format": "RGB888"}
                self.vc = Picamera2()
                video_config_cam = self.vc.create_video_configuration(main=cam_setup, controls={"FrameDurationLimits": (pifps, pifps)})
                self.vc.configure(video_config_cam)
                self.info(f"Camera resolution: {self.camera_resolution}, FPS: {self.camera_fps}")
            else:
                self.picam_feed = False
     
        except ValueError:
            self.stop_signal.set()
            self.reading_queue.put(None)
            self.error(f"Could not make VideoCapture from source '{video_source}'")

        self.info(
            f"Will sleep {self.sleep_seconds} seconds if reading queue fills up with {self.flush_thresh} frames. This *should not happen* if you're using a live webcam, else the frames are being processed too slowly!"
        )


    def run(self) -> None:
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
                grabbed, frame = self.vc.read()

            if frame is None or self.check_recording_complete(time_now):
                break

            if self.cam_feed is True:
                self.frame_count += 1
                if self.frame_count % 150 == 0:
                    self.info(f"Read {self.frame_count} frames so far. FPS: {self.calculate_fps(self.start_time, time_now)}")

            # Make sure queue has not filled up too much. This is really bad if
            # this happens for a *live* feed (i.e. webcam) since it means you
            # will lose the next `self.sleep_seconds` seconds of footage, but is
            # fine if working with video files as input
            if self.reading_queue.qsize() >= self.flush_thresh:
                self.debug(
                    f"Queue filled up to threshold. Sleeping {self.sleep_seconds} seconds to make sure queue can be drained..."
                )

                # Removing the smart sleep may increase the processing speeds.
                # self.sleep_seconds = self.smart_sleep(
                #     sleep_seconds=self.sleep_seconds, queue=self.reading_queue
                # )
                self.debug(
                    f"Finished sleeping with {self.reading_queue.qsize()} frames still in buffer!"
                )

            self.reading_queue.put(frame)

          
        # Append None to indicate end of queue
        self.info("Adding None to end of reading queue")
        self.reading_queue.put(None)
        if self.picam_feed is True:
            self.vc.stop()
            self.vc.close()
        else:
            self.vc.release()

        self.stop_signal.set()

    def get_fps(self) -> int:
        if self.picam_feed is True:
            return self.camera_fps
        else:
            return int(self.vc.get(cv2.CAP_PROP_FPS))
    
    def check_recording_complete(self, time_now) -> bool:
        if self.cam_feed and (time_now - self.start_time >= self.record_duration):
            return True
        else:
            return False
    
    def calculate_fps(self, start_time, end_time):
        return round(self.frame_count/(end_time-start_time),2)

    
    def get_frame_size(self) -> Tuple[int]:
        if self.picam_feed is True:
            return self.camera_resolution
        else:
            width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)

    @staticmethod
    def get_video_capture(source: Union[str, int]) -> cv2.VideoCapture:
        """
        Get a VideoCapture object from either a given filepath or an interger
        representing the index of a webcam (e.g. source=0). Raises a ValueError if
        we could not create a VideoCapture from the given `source`.

        :param source: a string representing a filepath for a video, or an integer
            representing a webcam's index.
        :return: a VideoCapture object for the given `source`.
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
        """
        Sleeps for `sleep_seconds` seconds. Afterwards, if `queue` is completely
        empty, this implies that we slept for too long, in which case this will
        return a number slightly *smaller* than `sleep_seconds` which should be
        used for the next sleep. If `queue` is still too full, then this will
        return a *larger* number. If `queue` is in sweet spot, then just returns
        the given `sleep_seconds`.

        This will never return a negative number (so long as you don't provide it
        with a negative number).

        :param queue: queue whose throughput we want to optimise
        :return: adjusted sleep time
        """
        qsize_before = queue.qsize()
        time.sleep(sleep_seconds)
        self.debug("WOKE UP!")
        qsize_after = queue.qsize()

        lower_qsize = int(0.05 * queue.maxsize)
        upper_qsize = int(0.15 * queue.maxsize)

        # Best case, we don't need to adust sleep time
        if qsize_after >= lower_qsize and qsize_after <= upper_qsize:
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; within range of {lower_qsize}-{upper_qsize}; sleep of {sleep_seconds} sec was ideal!"
            )
            return sleep_seconds

        # If queue size is *really* small, then reduce sleep by large amount
        if qsize_after <= 2:
            new_sleep_seconds = sleep_seconds * 0.90
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *below* range of {lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds

        # Need to shorten sleep time if too many frames are drained
        if qsize_after < lower_qsize:
            new_sleep_seconds = sleep_seconds * 0.95
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *below* range of {lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds

        # Need to extend sleep time if too few frames are drained
        if qsize_after > upper_qsize:
            new_sleep_seconds = sleep_seconds * 1.05
            LOGGER.debug(
                f"Queue size: {qsize_before} -> {qsize_after}; *above* range of {lower_qsize}-{upper_qsize}; adjusting sleep to {new_sleep_seconds:.2f} sec"
            )
            return new_sleep_seconds


class Writer(LoggingThread):
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
        frames_to_save: int
    ) -> None:
        super().__init__(name="WriterThread", logger=logger)

        self.writing_queue = writing_queue
        self.filepath = filepath
        self.frame_size = frame_size
        self.fps = fps
        self.stop_signal = stop_signal
        self.output_filename = output_filename
        self.embed_timestamps = embed_timestamps
        self.save_frames = save_frames
        if self.save_frames is True:
            self.frames_to_save = frames_to_save
        else:
            self.frames_to_save = 0


        self.fourcc = cv2.VideoWriter_fourcc(*video_codec)
        self.flush_thresh = int(0.25 * writing_queue.maxsize)
        self.info(
            f"Will flush buffer to output file every {self.flush_thresh} frames"
        )

        self.frame_count = 0
        self.recorded_frame_number = 0
        
    @classmethod
    def from_reader(
        cls,
        reader: Reader,
        writing_queue: Queue,
        filepath: str,
        stop_signal: Event,
        logger: logging.Logger,
        output_filename: str,
        video_codec: str,
        embed_timestamps: bool,
        save_frames: bool,
        frames_to_save: int
    ) -> Writer:
        """Convenience method to generate a Writer from a Reader.

        This is useful because the Writer should share the FPS and resolution
        of the input video as determined by the Reader. This just saves you
        having to parse those attributes yourself.

        Parameters
        ----------
        reader : Reader
            Reader whose 
        writing_queue : Queue
            Queue to retrieve video frames from. Some other thread should be 
            putting these frames into this queue for this Writer to retrieve.
        filepath : str
            Filepath for output video file.
        stop_signal : Event
            A threading Event that this Reader queries to know when to stop. 
            This is used for graceful termination of the multithreaded program.
        logger : logging.Logger
            Logger to use for logging key info, warnings, etc.

        Returns
        -------
        Writer
            Writer with same FPS and frame size as given Reader.
        """
        fps = reader.get_fps()
        frame_size = reader.get_frame_size()

        writer = Writer(
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
            frames_to_save=frames_to_save
        )
        return writer

    def run(self) -> None:
        vw = cv2.VideoWriter(
            filename=self.filepath,
            fourcc=self.fourcc,
            fps=self.fps,
            frameSize=self.frame_size,
        )
        saved_frames = 0
        frame_info = []
        loop_is_running = True
        currently_ommiting = False
        while loop_is_running:
            time.sleep(0.000001)
            
            

            if (
                self.writing_queue.qsize() < self.flush_thresh
                and not self.stop_signal.is_set()
            ):
                continue

            self.debug(
                f"Queue size exceeded ({self.writing_queue.qsize() >= self.flush_thresh}) OR stop signal ({self.stop_signal.is_set()})"
            )

            # Only flush the threshold number of frames, OR remaining frames if there are only a few lef
            frames_to_flush = min(self.writing_queue.qsize(), self.flush_thresh)
            # frames_to_flush = self.writing_queue.qsize()
            # print(frames_to_flush, "Writer----------------")
            self.debug(f"Flushing {frames_to_flush} frames...")
            

            for i in range(frames_to_flush):
            # if self.writing_queue.qsize() > 0:
                try:
                    frame_combo= self.writing_queue.get(timeout=10)
                except Empty:
                    self.warning(f"Waited too long for frame! Exiting...")
                    loop_is_running = False
                    break

                # print(self.frame_count, "Writer")

                if frame_combo is None:
                    loop_is_running = False
                    break
                else:
                    frame, nframe, first_in_seq ,ff_recorded = frame_combo

                self.frame_count += 1

                if self.embed_timestamps is True:
                    raw_time = nframe / (self.fps)
                    minutes, seconds = divmod(raw_time, 60)
                    time_str = f"{int(minutes):02d}:{int(seconds):02d}"
                    cv2.putText(
                        frame,
                        text=f"Frame: {self.frame_count}  |  Raw Frame: {nframe}  |  Time: {time_str}",
                        org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2,
                    )

                
                if self.save_frames is True and saved_frames < self.frames_to_save:
                    image_filepath = Path(self.filepath).parent / f"{self.output_filename}_frame_{self.frame_count}.jpg"
                    cv2.imwrite(str(image_filepath), frame)
                    saved_frames += 1
                    


                vw.write(frame)

                if first_in_seq is True:
                    if ff_recorded is True:
                        nff_number = self.frame_count
                    else:
                        nff_number = None

                    frame_info.append([self.frame_count, nframe, nff_number])


                if self.frame_count % 50 == 0:
                    self.info(f"Written {self.frame_count} frames so far")
            self.debug(f"Flushed {frames_to_flush} frames!")

        vw.release()

        # Write CSV file with omitted frame indices. Note this is not the most
        # space-efficient way to store these, but it's probs good enough
        csv_filepath = Path(self.filepath).parent / f"{self.output_filename}_video_info.csv"
        with open(csv_filepath, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["frame_number", "original_frame_number", "frame_with_full_frame"])
            for row in frame_info:
                csv_writer.writerow(row)

    



class MotionDetector(LoggingThread):
    buffer_count = 0
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
        super().__init__(name="MotionThread", logger=logger)

        self.input_queue = input_queue
        self.writing_queue = writing_queue
        self.stop_signal = stop_signal
        self.prev_frame = None
        self.nframe = 0


        # For motion detection
        self.downscale_factor = downscale_factor
        self.fx = self.fy = 1 / downscale_factor
        downscaled_kernel_size = int(dilate_kernel_size / self.downscale_factor)
        self.dilation_kernel = np.ones((downscaled_kernel_size, downscaled_kernel_size))
        self.movement_threshold = movement_threshold
        self.post_motion_record_frames = post_motion_record_frames
        self.full_frame_capture_interval  = full_frame_capture_interval
        self.background_transparency = background_transparency
        if self.background_transparency > 0:
            self.transparent_background = True
        else:
            self.transparent_background = False

        if self.post_motion_record_frames > 0:
            self.buffer_analysis = True

        if self.downscale_factor != 1:
            self.info(
                f"Dilation kernel downscaled by {downscale_factor}x from {dilate_kernel_size} to {self.dilation_kernel.shape[0]}"
            )

    def run(self) -> None:
        self.prev_frame = None
        self.prev_mask = None
        self.buffer_analysis = True

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

        # Make sure motion writer knows to stop
        self.writing_queue.put(None)


    def detect_motion(self, frame):
        self.nframe += 1
        put_to_queue = False
        first_frame_in_seq = False

        # Downscale input frame to improve performance
        orig_shape = frame.shape
        
        full_frame_recorded = False

        if (self.nframe % self.full_frame_capture_interval == 0) or (self.nframe == 1):
            self.record_full_frame = True


        if self.downscale_factor == 1:
            # Downscale factor of 1 really just means no downscaling at all
            downscaled_frame = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
        else:
            downscaled_frame = cv2.cvtColor(cv2.resize(frame, dsize=None, fx=self.fx, fy=self.fy),  cv2.COLOR_BGR2GRAY) # what if gray scale and then resize
            

        if self.prev_frame is None:
            self.prev_frame = downscaled_frame

        # Compute pixel difference between consecutive frames 
        diff = cv2.absdiff(downscaled_frame, self.prev_frame)

        # Convert to grayscale
        gray_frame = cv2.dilate(diff, kernel=self.dilation_kernel)


        # Cut off pixels that did not have "enough" movement. This is now a 2D array of just 1s and 0s
        _, threshed_diff = cv2.threshold(src=gray_frame, thresh=self.movement_threshold, maxval=255, type=cv2.THRESH_BINARY)

        mask = cv2.medianBlur(cv2.dilate(threshed_diff, kernel=self.dilation_kernel), 9)


        if self.prev_mask is None:
            self.prev_mask = mask.copy()


        # Check if there are any nonzero pixels in the mask.
        if cv2.countNonZero(mask):

            if self.downscale_factor != 1:
                mask = cv2.resize(mask, dsize=(orig_shape[1], orig_shape[0]))
                
            if self.transparent_background is True:
                transparent_frame = cv2.addWeighted(frame, (self.background_transparency), np.zeros_like(frame), (1-self.background_transparency), 0)

            if not cv2.countNonZero(self.prev_mask):
                first_frame_in_seq = True

                if self.record_full_frame is True:
                    motion_frame = frame.copy()
                    self.record_full_frame = False
                    full_frame_recorded = True
                else:
                    motion_frame = cv2.bitwise_and(frame, frame, mask=mask)
                    if self.transparent_background is True:
                        motion_frame = cv2.add(cv2.absdiff(motion_frame, transparent_frame), motion_frame)

            else:
                motion_frame = cv2.bitwise_and(frame, frame, mask=mask)
                if self.transparent_background is True:
                    motion_frame = cv2.add(cv2.absdiff(motion_frame, transparent_frame), motion_frame)
            
            
            self.prev_mask = mask.copy()
            self.buffer_count = 0
            put_to_queue = True

        elif self.buffer_analysis is True and cv2.countNonZero(self.prev_mask):
            if self.buffer_count < self.post_motion_record_frames:
                self.buffer_count += 1
                motion_frame = cv2.bitwise_and(frame, frame, mask=self.prev_mask)
                put_to_queue = True

            else:
                self.prev_mask = mask.copy()

        else:
            self.prev_mask = mask.copy()

        if put_to_queue is True:
            self.writing_queue.put([motion_frame, self.nframe, first_frame_in_seq  ,full_frame_recorded])

        self.prev_frame = downscaled_frame.copy()


def main(config: Config):
    start = time.time()

    # Make sure opencv doesn't use too many threads and hog CPUs
    cv2.setNumThreads(config.num_opencv_threads)

    # Use the input filepath to figure out the output filename
    if type(config.video_source) is str:
        output_filename = os.path.splitext(os.path.basename(config.video_source))[0]
    else:
        output_filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine the output directory based on user input

    if os.path.isdir(config.output_directory):
        output_parent_directory = Path(config.output_directory, "EcoMotionZip")
        log_message = f"Outputting to {output_parent_directory}"
    else:
        if os.path.isdir(config.video_source):
            output_parent_directory = Path(config.video_source, "EcoMotionZip")
            log_message =f"Output directory not specified or unavailable. Outputting to video source directory {output_parent_directory}"
        else:
            output_parent_directory = Path(Path(config.video_source).parent, "EcoMotionZip")
            log_message = f"Output directory not specified or unavailable. Outputting to video source directory  {output_parent_directory}"



    # output_parent_directory = Path(config.output_directory , "EcoMotionZip")
    if not output_parent_directory.exists():
        os.makedirs(output_parent_directory, exist_ok=True)

    # Create queues for transferring data between threads (or processes)
    reading_queue = Queue(maxsize=512)
    writing_queue = Queue(maxsize=256)

    stop_signal = Event()

    output_directory = Path(f"{output_parent_directory}/{output_filename}")
    # output_directory = Path(f"out/{output_filename}")
    if not output_directory.exists():
        output_directory.mkdir()
    output_filepath = str(output_directory / f"{output_filename}.mp4s")

    # Create some handlers for logging output to both console and file
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(threadName)-14s] %(msg)s"))
    file_handler = logging.FileHandler(filename=output_directory / f"{output_filename}_output.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(threadName)-14s] %(msg)s")
    )
    # Make sure any prior handlers are removed
    LOGGER.handlers.clear()
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)

    # formatted_start_time  = start.strftime("%Y-%m-%d %H:%M:%S")
    LOGGER.info(f"Starting processing at :  {datetime.fromtimestamp(start)}")
    LOGGER.info(f"Running main() with Config:  {config.__dict__}")
    LOGGER.info(f"Outputting to {log_message}")

    # Create all of our threads
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
            save_frames= config.save_frames,
            frames_to_save=config.frames_to_save,
            embed_timestamps=config.embed_timestamps,
        ),
    )

    for thread in threads:
        LOGGER.info(f"Starting {thread.name}")
        thread.start()

    # Regularly poll to check if all threads have finished. If they haven't finished,
    # just sleep a little and check later
    while True:
        try:
            time.sleep(0.5)
            if not any([thread.is_alive() for thread in threads]):
                LOGGER.info(
                    "All child processes appear to have finished! Exiting infinite loop..."
                )
                break

            for queue, queue_name in zip(
                [reading_queue, writing_queue],
                ["Reading", "Writing"],
            ):
                LOGGER.debug(f"{queue_name} queue size: {queue.qsize()}")
        except (KeyboardInterrupt, Exception) as e:
            LOGGER.exception(
                "Received KeyboardInterrupt or some kind of Exception. Setting interrupt event and breaking out of infinite loop...",
            )
            LOGGER.warning(
                "You may have to wait a minute for all child processes to gracefully exit!",
            )
            stop_signal.set()
            break

    for thread in threads:
        LOGGER.info(f"Joining {thread.name}")
        thread.join()

    # Add any extra stats/metadata to output too
    end = time.time()
    duration_seconds = end - start
    # formatted_end_time = end.strftime("%Y-%m-%d %H:%M:%S")
    LOGGER.info(f"Finished processing at :  {datetime.fromtimestamp(end)}")
    LOGGER.info(f"Finished main() in {duration_seconds:.2f} seconds.")

    if type(config.video_source) is str and config.delete_original_after_processing is True:
        os.remove(config.video_source)
        LOGGER.info(f"Deleted original video file: {config.video_source}")
    


if __name__ == "__main__":
    downscale_factor = CONFIG.downscale_factor
    dilate_kernel_size = CONFIG.dilate_kernel_size
    movement_threshold = CONFIG.movement_threshold
    post_motion_record_frames = CONFIG.post_motion_record_frames
    full_frame_capture_interval = CONFIG.full_frame_capture_interval
    video_codec = CONFIG.video_codec
    video_source = CONFIG.video_source
    embed_timestamps = CONFIG.embed_timestamps
    
    if type(video_source) != int:
        video_source = Path(video_source)
        if video_source.is_dir():
            video_source = [str(v) for v in video_source.iterdir() if v.suffix in ['.avi', '.mp4', '.h264', '.MTS']]
        elif type(video_source) is not list:
            video_source = [str(video_source)]
    else:
        # Just to make it iterable
        video_source = [video_source]*CONFIG.number_of_videos


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
    # print("Length of parameter_combos:", len(parameter_combos))

    for combo in parameter_combos:
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
