
# EcoMotionZip


 * EcoMotionZip is a software project that provides a set of packages for efficient and reliable data compression and decompression of motion capture data.
 * The EcoMotionZip software packages utilize advanced compression algorithms and techniques to reduce the storage space required for motion capture data while maintaining high fidelity and accuracy during decompression.
 * This software project aims to improve the efficiency and effectiveness of motion capture data storage and transmission in various applications in video camera trap based animal monitoring in resource constrained edge devices.
 

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





<!-- ## Installation

To install EcoMotionZip, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/yourusername/EcoMotionZip.git
```

2. Navigate to the cloned directory:

```bash
cd EcoMotionZip
```

3. Create a virtual environment:

```bash
python3 -m venv env
```

4. Activate the virtual environment:

On Windows:

```bash
.\env\Scripts\activate
```

On Unix or MacOS:

```bash
source env/bin/activate
```

5. Install the dependencies:

```bash
pip install -r requirements.txt
``` -->

## Usage

Here's a quick example of how to use EcoMotionZip:

```python
# Python code example
```

## Contributing

We welcome contributions to EcoMotionZip! Please see our [contributing guide](CONTRIBUTING.md) for more details.

## License

EcoMotionZip is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions, feel free to reach out to us at [email](mailto:example@example.com).
```
