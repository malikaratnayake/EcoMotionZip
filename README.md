
# EcoMotionZip
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://choosealicense.com/licenses/gpl-3.0/)


## Overview

EcoMotionZip is an open-source Python-based software developed for motion-based video compression and analysis. It is optimised for efficient and reliable video data compression on resource-constrained camera traps and desktop systems. By analysing motion data captured by video camera traps, EcoMotionZip selectively retains video segments with motion, enabling precise animal behaviour analysis through both automated and manual methods.

EcoMotionZip serves as a robust and flexible tool for researchers and practitioners working with video data, offering advanced capabilities to streamline workflows in video analysis and compression.

## Variants

EcoMotionZip is available in two versions:

1. **EcoMotionZip**: Designed for desktop systems, this version includes a user-friendly Graphical User Interface (GUI) for easy configuration and usage.
2. **EcoMotionZip Lite**: A command-line version tailored for resource-constrained devices, ensuring lightweight and efficient operation.

## Key Features

#### 1. Motion-Based Video Compression for Resource-Constrained Devices
EcoMotionZip has been tested on devices such as Raspberry Pi microcomputers and NVIDIA Jetson platforms, demonstrating its ability to compress recorded videos effectively while minimising resource usage.

#### 2. Real-Time Video Compression and Capture
EcoMotionZip facilitates real-time video capture and compression using Raspberry Pi systems integrated with the PiCamera2 library. This allows single-pass operation, enabling immediate video compression during capture.

#### 3. Compatibility with Polytrack for Automated Insect Tracking
Videos compressed by EcoMotionZip can be directly processed by **Polytrack** to extract spatiotemporal details of insect movement, including the reconstruction of trajectories.

#### 4. Optimisation for Manual Behavioural Observations
EcoMotionZip enhances camera trap videos for manual analysis by:
   - Removing inactive video segments.
   - Highlighting pixel regions with detected motion to draw the observer’s focus to relevant areas.

#### 5. Frame Extraction for AI Training
EcoMotionZip enables the extraction of frames containing motion, saving them as images for creating datasets suitable for training Convolutional Neural Networks (CNNs) and other AI models.



 

## Installation and Dependencies

EcoMotionZip relies on several essential packages for its basic functionality, including Numpy, OpenCV, and FFMPEG. This documentation provides step-by-step instructions for setting up the EcoMotionZip software on a Raspberry Pi platform running the BookWorm OS. It is compatible with Raspberry Pi or similar Linux-based platforms, whether or not a virtual environment is used.

**1. Update and Upgrade Packages**
   
Before installing EcoMotionZip, ensure your system packages are up-to-date by running the following commands:
```bash
sudo apt update
sudo apt upgrade
```


**2. Install OpenCV**

Install the latest version of OpenCV:

```bash
sudo apt install python3-opencv
```

**3. Install PiCamera2 Support (Optional):**

If you intend to run EcoMotionZip in real-time using Raspberry Pi Camera V3 or later, install the libcamera package to enable PiCamera2 support.

```bash
sudo apt install -y python3-libcamera python3-kms++ libcap-dev
```

**4. Install Additional Codecs with FFMPEG**

Ensure proper codec support for FFMPEG:
```bash
sudo apt-get install ffmpeg x264 libx264-dev
```
**5. Install Git Support**

Install Git to clone the EcoMotionZip repository from GitHub:
```bash
sudo apt install git
```
**6. Clone EcoMotionZip from GitHub**

Clone the EcoMotionZip package to your local environment:
```bash
git clone https://github.com/yourusername/EcoMotionZip.git
```

### Tested Dependencies
EcoMotionZip has been tested with the following versions of dependencies:

- Python: 3.11.2
- Numpy: 1.24.2
- opencv-python: 4.6.0
- FFMPEG: version 5.1.4
  
Ensure that your system matches these versions for optimal performance. If you encounter any issues, refer to the troubleshooting section or consult the EcoMotionZip GitHub repository for additional support and updates.

> PyPi EcoMotionZip package coming soon!

## Usage

EcoMotionZip software can be used in both offline and real-time to process camera trap videos. We recommend processing videos in offline mode depending on the specification of your edge computing platform.

The processing parameters for EcoMotionZip can be set through `config.json` file or as commanline arguments. An example of `config.json` file and a complete set of parameters with description are presented below.

Please follow the following steps to run the EcoMotionZip after clonning the repository.

1. Navigate to the cloned directory.
   ```bash
   cd EcoMotionZip
   ```

2. Run EcoMotionZip software.
   
   ```bash
   python EcomotionZip/arc.py
   ```
#### Example of the `config.json` file

```json
{
    "video_source": "/path/to/video/directory",
    "output_directory": "/path/to/output/directory",
    "record_duration": 60,
    "number_of_videos": 1,
    "camera_resolution": [1920,1080],
    "camera_fps": 30,
    "raspberrypi_camera": false,
    "delete_original": false,
    "reader_sleep_seconds": 1,
    "reader_flush_proportion": 0.9,
    "downscale_factor": 16,
    "dilate_kernel_size": 128,
    "movement_threshold": 40,
    "persist_frames": 0,
    "full_frame_guarantee": 300,
    "video_codec": "X264",
    "num_opencv_threads": 10
}
```

### List of EcoMotionZip parameters and usage
- `-h, --help`        
  show this help message and exit

- `--video_source VIDEO_SOURCE`     
  Path to the input directory or a single video file. Set value to 0 to use webcam or any other integer to use a different camera.
- `--output_directory OUTPUT_DIRECTORY`   
  Path to the output directory
- `--record_duration RECORD_DURATION`   
  Duration of the recording for a single video in seconds.
- `--number_of_videos NUMBER_OF_VIDEOS`   
Number of videos to record.
- `--camera_resolution CAMERA_RESOLUTION`   
  Resolution of the camera.
- `--camera_fps CAMERA_FPS`    
  FPS of the camera.
- `--delete_original DELETE_ORIGINAL`   
 Delete original video after processing.
-  `--downscale_factor DOWNSCALE_FACTOR`  
   Downscale factor for input video.
- `--dilate_kernel_size DILATE_KERNEL_SIZE`   
Kernel size for dilation.
- `--movement_threshold MOVEMENT_THRESHOLD`     
Threshold for movement detection.
- `--persist_frames PERSIST_FRAMES`      
Number of frames to persist for.
- `--full_frame_guarantee FULL_FRAME_GUARANTEE`        
Number of frames to persist for.
- `--video_codec {XVID,X264}`          
Video codec to use for output video.
- `--num_opencv_threads NUM_OPENCV_THREADS`     
    Number of threads to use for OpenCV.

## License

EcoMotionZip is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions, feel free to reach out to us at [email](mailto:malika.ratnayake@monash.edu).

## References

* [Bees-edge](https://github.com/byebrid/bees-edge) by [Lex Gallon](https://github.com/byebrid).
* [Basic motion detection and tracking with Python and OpenCV](https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/) by [pyimagesearch](https://pyimagesearch.com)
* [Increasing webcam FPS with Python and OpenCV](https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/) by [pyimagesearch](https://pyimagesearch.com)




