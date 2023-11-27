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

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

class Config:
    """Configuration class just to provide better parameter hints in code editor."""
    def __init__(
        self,
        video_source: str,
        reader_sleep_seconds: float,
        reader_flush_proportion: float,
        downscale_factor: int,
        dilate_kernel_size: int,
        movement_threshold: int,
        persist_frames: float,
        video_codec: str,
        num_opencv_threads: int,
    ) -> None:
        self.video_source = video_source
        self.reader_sleep_seconds = reader_sleep_seconds
        self.reader_flush_proportion = reader_flush_proportion
        self.downscale_factor = downscale_factor
        self.dilate_kernel_size = dilate_kernel_size
        self.movement_threshold = movement_threshold
        self.persist_frames = persist_frames
        self.video_codec = video_codec
        self.num_opencv_threads = num_opencv_threads


# Create Config object from JSON file
with open("config.json", "r") as f:
    __config_dict = json.load(f)
    CONFIG = Config(**__config_dict)


class LoggingThread(Thread):
    """A wrapper around `threading.Thread` with convenience methods for logging."""
    def __init__(self, name: str, logger: logging.Logger) -> None:
        super().__init__(name=name)

        self.logger = logger

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)


class Reader(LoggingThread):
    """A class to read video from either a file or camera, and push to a queue.
    
    Reads video from file or camera and pushes each frame onto a queue. Note that
    this queue *must* be emptied before the the program can be closed, meaning
    every frame in this queue needs to be handled by some other thread, either
    with some actual processing or just popping those frames and doing nothing 
    with them.

    This Reader includes "smart sleeping". This is intended for use with video 
    files, where there is no real upper bound on the rate at which we read frames
    (as opposed to a live camera feed, where you'll be bounded by its FPS). For 
    files, this Reader can read frames much more quickly than they can be 
    processed, meaning the reading queue can fill up and start blocking this 
    reading thread when it tries to push frames onto the full queue (and this 
    blocking still uses the CPU meaning it's a waste of compute time!). We 
    therefore let this thread sleep for a while once its queue fills above some
    threshold (e.g. ~90%).
    """
    def __init__(
        self,
        reading_queue: Queue,
        video_source: Union[str, int],
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
        self.stop_signal = stop_signal
        self.sleep_seconds = sleep_seconds
        self.video_file_name = os.path.basename(video_source)
        
        self.flush_thresh = int(flush_proportion * reading_queue.maxsize)
        # Make video capture now so we can dynamically retrieve its FPS and frame size
        try:
            self.vc = self.get_video_capture(source=self.video_source)
        except ValueError:
            self.stop_signal.set()
            self.reading_queue.put(None)
            self.error(f"Could not make VideoCapture from source '{video_source}'")

        self.info(
            f"Will sleep {self.sleep_seconds} seconds if reading queue fills up with {self.flush_thresh} frames. This *should not happen* if you're using a live webcam, else the frames are being processed too slowly!"
        )

    def run(self) -> None:
        while True:
            if self.stop_signal.is_set():
                self.info("Received stop signal")
                break

            grabbed, frame = self.vc.read()
            if not grabbed or frame is None:
                break

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
        self.vc.release()
        self.stop_signal.set()

    def get_fps(self) -> int:
        return int(self.vc.get(cv2.CAP_PROP_FPS))

    
    def get_frame_size(self) -> Tuple[int]:
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
            return cv2.VideoCapture(filename=source)
        elif type(source) is int:
            return cv2.VideoCapture(index=source)
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
    ) -> None:
        super().__init__(name="WriterThread", logger=logger)

        self.writing_queue = writing_queue
        self.filepath = filepath
        self.frame_size = frame_size
        self.fps = fps
        self.stop_signal = stop_signal
        self.output_filename = output_filename

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
        )
        return writer

    def run(self) -> None:
        vw = cv2.VideoWriter(
            filename=self.filepath,
            fourcc=self.fourcc,
            fps=self.fps,
            frameSize=self.frame_size,
        )
        
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


                vw.write(frame)
                self.frame_count += 1
                

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
        output_dir = Path(self.filepath).parent
        csv_filepath = output_dir / f"{self.output_filename}_video_info.csv"
        with open(csv_filepath, "w") as f:
            csv_writer = csv.writer(f)
            for row in frame_info:
                csv_writer.writerow(row)



class MotionDetector(LoggingThread):
    insect_detected = False
    use_deep_learning = False
    dl_attempt =0
    buffer_count = 0
    max_buffer = 0 
    dl_run_freq = 1 # Set this to 1 to run deep learning every frame (when prompted)
    def __init__(
        self,
        input_queue: Queue,
        writing_queue: Queue,
        downscale_factor: int,
        dilate_kernel_size: int,
        movement_threshold: int,
        persist_frames: float,
        stop_signal: Event,
        logger: logging.Logger,
    ) -> None:
        super().__init__(name="MotionThread", logger=logger)

        self.input_queue = input_queue
        self.writing_queue = writing_queue
        self.stop_signal = stop_signal

        self.prev_frame = None
        self.nframe = 0
        self.full_frame = True
        # self.prev_diff = None


        # For motion detection
        self.downscale_factor = downscale_factor
        self.fx = self.fy = 1 / downscale_factor
        downscaled_kernel_size = int(dilate_kernel_size / self.downscale_factor)
        self.dilation_kernel = np.ones((downscaled_kernel_size, downscaled_kernel_size))
        self.movement_threshold = movement_threshold
        self.persist_frames = persist_frames

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

            # motion_detected_frame = self.detect_motion(frame=frame)
            self.detect_motion(frame=frame)


            # self.writing_queue.put(motion_detected_frame)

        # Make sure motion writer knows to stop
        self.writing_queue.put(None)

    def run_deep_learning(self, _frame, _mask, _conf = 0.01) -> bool:
        # Find contours in the mask
        contours, _ = cv2.findContours(image=_mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            dl_frame = _frame[y:y + h, x:x + w]
            dl_frame_size = dl_frame.shape[0:2]
            # Round off values in dl_frame_size to nearest multiple of 32
            dl_frame_size = tuple([32 * round(x / 32) for x in dl_frame_size])

            results = model.predict(source=dl_frame, conf=_conf, imgsz = dl_frame_size ,show=False, verbose = False)

            if results[0]:
                return True
            else:
                return False
        else:
            return False


    def detect_motion(self, frame):
        self.nframe += 1
        # print(self.nframe, "Motion")
        # Downscale input frame
        height, width, _ = frame.shape
        frame = frame[:, int((width - height) / 2) - 400 : int((width + height) / 2) - 400, :]
        orig_shape = frame.shape
        
        full_frame_recorded = False
        # channels= cv2.split(frame)

        if self.nframe <= 5:
            # Crop the frame from left and right to make it square
            # height, width, _ = frame.shape

            # frame0 = frame[:, int((width - height) / 2) : int((width + height) / 2), :]

            cv2.imwrite(f"out/full_frame-{self.nframe}.jpg", frame)



        if (self.nframe % 300 == 0) or (self.nframe == 1):
            self.record_full_frame = True



        if self.downscale_factor == 1:
            # Downscale factor of 1 really just means no downscaling at all
            downscaled_frame = frame
        else:
            # downscaled_frame = cv2.cvtColor(cv2.resize(frame, dsize=None, fx=self.fx, fy=self.fy),  cv2.COLOR_BGR2GRAY) # what if gray scale and then resize
            downscaled_frame0 = cv2.resize(frame, dsize=None, fx=self.fx, fy=self.fy)
            downscaled_frame = cv2.cvtColor(downscaled_frame0,  cv2.COLOR_BGR2GRAY)

        if self.nframe == 5:
            cv2.imwrite("out/downscaled_frame.jpg", downscaled_frame)
            cv2.imwrite("out/prev_frame.jpg", self.prev_frame)
            cv2.imwrite("out/downscaled_frame0.jpg", downscaled_frame0)
            

        if self.prev_frame is None:
            self.prev_frame = downscaled_frame
        # if self.prev_diff is None:
        #     self.prev_diff = np.zeros(self.prev_frame.shape, dtype=np.uint8)

        # Compute pixel difference between consecutive frames (note this still has 3 channels)
        # Note that previous frame was already downscaled!
        diff = cv2.absdiff(downscaled_frame, self.prev_frame)
        # Add decayed version of previous diff to help temporarily stationary bees "persist"
        # diff += (self.prev_diff * self.persist_frames).astype(np.uint8)

        if self.nframe == 5:
            cv2.imwrite("out/diff_frame.jpg", diff)
        

    



        # Convert to grayscale
        # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.dilate(diff, kernel=self.dilation_kernel)

        if self.nframe == 5:
            cv2.imwrite("out/gray_frame.jpg", gray_frame)

        # Cut off pixels that did not have "enough" movement. This is now a 2D array
        # of just 1s and 0s
        _, threshed_diff = cv2.threshold(src=gray_frame, thresh=self.movement_threshold, maxval=255, type=cv2.THRESH_BINARY)

        if self.nframe == 5:
            cv2.imwrite("out/threshed_diff.jpg", threshed_diff)


        # Go to previous setting, set the movement threshold to a minimum valuyes

        # Filter the blobs in threshed_diff based on a minimum size
        # threshed_diff = threshed_diff.astype(np.uint8)

        # mask = cv2.medianBlur(cv2.dilate(threshed_diff, kernel=self.dilation_kernel), 9)
        mask0 = cv2.dilate(threshed_diff, kernel=self.dilation_kernel)
        mask = cv2.medianBlur(mask0, 9)

        if self.nframe == 5:
            cv2.imwrite("out/mask.jpg", mask)
            cv2.imwrite("out/dilate_thres.jpg", mask0)


        if self.prev_mask is None:
            self.prev_mask = mask.copy()

        if cv2.countNonZero(mask):
            record_frame = True

            if not cv2.countNonZero(self.prev_mask):
                first_frame_in_seq = True
            else:
                if self.insect_detected is True:
                    first_frame_in_seq = False
                else:
                    first_frame_in_seq = True

            self.prev_mask = mask.copy()
            self.buffer_count = 0

        else:
            record_frame = False
            first_frame_in_seq = False
            if self.insect_detected is True and self.buffer_count > self.max_buffer:
                self.insect_detected = False
                self.prev_mask = mask.copy()
            else:
                mask = self.prev_mask.copy()

            self.buffer_count += 1

                

        if self.downscale_factor != 1:
            mask = cv2.resize(mask, dsize=(orig_shape[1], orig_shape[0]))

        # Save downscaled frame for use in next iteration
        self.prev_frame = downscaled_frame.copy()

        
        motion_frame = np.zeros(shape=frame.shape, dtype=np.uint8)

        if record_frame is True:   
            
            if first_frame_in_seq is True:
                if self.use_deep_learning is True and (self.dl_attempt % self.dl_run_freq == 0):
                    self.insect_detected = self.run_deep_learning(_frame=frame, _mask=mask, _conf=0.05)
                    if self.insect_detected is True:
                        self.dl_attempt = 0
                    else:
                        self.dl_attempt += 1
                elif self.use_deep_learning is True and (self.dl_attempt % self.dl_run_freq != 0):
                    self.dl_attempt += 1
                else:
                    self.insect_detected = True
                    self.dl_attempt = 0

                if self.record_full_frame is True and self.insect_detected is True:
                    # Copy the frame to motion_frame using opencv
                    motion_frame = frame.copy()

                    # motion_frame = frame
                    self.record_full_frame = False
                    full_frame_recorded = True
                else:
                    # motion_frame[mask] = frame[mask]
                    motion_frame = cv2.bitwise_and(frame, frame, mask=mask)

            elif self.insect_detected is True:
                motion_frame = cv2.bitwise_and(frame, frame, mask=mask)

            else:
                self.insect_detected = False

            put_to_queue = [motion_frame, self.nframe, first_frame_in_seq  ,full_frame_recorded]

            if self.nframe == 5:
                cv2.imwrite("out/motion_frame.jpg", motion_frame)
                cv2.imwrite("out/resided_mask.jpg", mask)

            self.writing_queue.put(put_to_queue)

        else:
            pass

        # return [motion_frame, record_frame ,full_frame_recorded]


def main(config: Config):
    start = time.time()

    # Make sure opencv doesn't use too many threads and hog CPUs
    cv2.setNumThreads(config.num_opencv_threads)

    # Use the input filepath to figure out the output filepath
    output_filename = os.path.splitext(os.path.basename(config.video_source))[0]

    # Create queues for transferring data between threads (or processes)
    reading_queue = Queue(maxsize=512)
    # motion_input_queue = Queue(maxsize=900) # previously 512
    writing_queue = Queue(maxsize=256)

    stop_signal = Event()

    # Figure out output filepath for this particular run
    # output_directory = Path(f"out/{datetime.now()}")
    output_directory = Path(f"out/{output_filename}")
    if not output_directory.exists():
        output_directory.mkdir()
    output_filepath = str(output_directory / f"{output_filename}.avi")

    # Copy config for record-keeping
    with open(output_directory / f"{output_filename}_config.json", "w") as f:
        # print(config.__dict__)
        json.dump(config.__dict__, f)

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

    LOGGER.info(f"Running main() with Config:  {config.__dict__}")
    LOGGER.info(f"Outputting to {output_filepath}")

    # filename_ = os.path.splitext(os.path.basename(config.video_source))[0]

    # Create all of our threads
    threads = (
        reader := Reader(
            reading_queue=reading_queue,
            video_source=config.video_source,
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
            persist_frames=config.persist_frames,
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
    stats = {"duration_seconds": round(duration_seconds, 2)}
    with open(output_directory / "stats.json", "w") as f:
        json.dump(stats, f)

    LOGGER.info(f"Finished main() in {duration_seconds:.2f} seconds.")


if __name__ == "__main__":
    downscale_factor = CONFIG.downscale_factor
    dilate_kernel_size = CONFIG.dilate_kernel_size
    movement_threshold = CONFIG.movement_threshold
    persist_frames = CONFIG.persist_frames
    video_codec = CONFIG.video_codec

    # Figure out if video is webcam index, single video file, or directory of
    # video files
    video_source = CONFIG.video_source
    if type(video_source) != int:
        video_source = Path(video_source)
        if video_source.is_dir():
            video_source = [str(v) for v in video_source.iterdir()]
        elif type(video_source) is not list:
            video_source = [str(video_source)]
    else:
        # Just to make it iterable
        video_source = [video_source]

    if type(downscale_factor) is not list:
        downscale_factor = [downscale_factor]
    if type(dilate_kernel_size) is not list:
        dilate_kernel_size = [dilate_kernel_size]
    if type(movement_threshold) is not list:
        movement_threshold = [movement_threshold]
    if type(persist_frames) is not list:
        persist_frames = [persist_frames]
    if type(video_codec) is not list:
        video_codec = [video_codec]

    parameter_combos = product(
        video_source,
        downscale_factor,
        dilate_kernel_size,
        movement_threshold,
        persist_frames,
        video_codec,
    )
    parameter_keys = [
        "video_source",
        "downscale_factor",
        "dilate_kernel_size",
        "movement_threshold",
        "persist_frames",
        "video_codec",
    ]
    for combo in parameter_combos:
        this_config_dict = dict(zip(parameter_keys, combo))
        this_config_dict.update(
            {
                "reader_sleep_seconds": CONFIG.reader_sleep_seconds,
                "reader_flush_proportion": CONFIG.reader_flush_proportion,
                "num_opencv_threads": CONFIG.num_opencv_threads,
                "video_codec": CONFIG.video_codec,
            }
        )
        this_config = Config(**this_config_dict)
        main(this_config)
