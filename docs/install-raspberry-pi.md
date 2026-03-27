# Installing EcoMotionZip — Raspberry Pi / Headless

This guide is for running EcoMotionZip **without a graphical interface** — on a Raspberry Pi,
a remote server, or any Linux system. No screen or desktop environment is required.

Tested on: Raspberry Pi OS Bookworm, Python 3.11.2, OpenCV 4.6.0, FFmpeg 5.1.4

---

## Step 1 — Update system packages

```bash
sudo apt update && sudo apt upgrade -y
```

---

## Step 2 — Install OpenCV

```bash
sudo apt install -y python3-opencv python3-numpy
```

---

## Step 3 — Install FFmpeg and H.264 codec support

Required for X264 and other codec options:

```bash
sudo apt install -y ffmpeg x264 libx264-dev
```

---

## Step 4 — (Optional) PiCamera2 support

Only needed if you are using a Raspberry Pi Camera Module for live capture:

```bash
sudo apt install -y python3-libcamera python3-kms++ libcap-dev python3-picamera2
```

---

## Step 5 — Install Git and clone EcoMotionZip

```bash
sudo apt install -y git
git clone https://github.com/malikaratnayake/EcoMotionZip.git
cd EcoMotionZip
```

---

## Step 6 — Run EcoMotionZip

Process a video file:

```bash
python run_headless.py --video_source /path/to/video.mp4 --output_directory /path/to/output
```

Process all videos in a folder:

```bash
python run_headless.py --video_source /path/to/folder --output_directory /path/to/output
```

Live capture with Raspberry Pi camera:

```bash
python run_headless.py --output_directory /path/to/output
```
*(Set `"raspberrypi_camera": true` in `config.json` first)*

See all available options:

```bash
python run_headless.py --help
```

---

## Configuration

Edit `config.json` in the EcoMotionZip folder to set your defaults:

```json
{
    "video_source": "",
    "output_directory": "",
    "video_codec": "X264",
    "raspberrypi_camera": false,
    "movement_threshold": 40,
    "downscale_factor": 16,
    "background_transparency": 0.0,
    "save_frames": false,
    "embed_timestamps": false,
    "delete_original_after_processing": false
}
```

Any value in `config.json` can be overridden on the command line.
For example: `python run_headless.py --movement_threshold 25 --video_codec FFV1`

---

## Running automatically on startup (optional)

To run EcoMotionZip automatically when the Pi boots, add a cron job:

```bash
crontab -e
```

Add this line (adjust paths as needed):

```
@reboot cd /home/pi/EcoMotionZip && python run_headless.py >> /home/pi/ecomotioinzip.log 2>&1
```

---

## Updating EcoMotionZip

```bash
cd EcoMotionZip
git pull
```
