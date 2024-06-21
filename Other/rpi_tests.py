import os
import time

#output_directory = './testing/Rep1'
video_codec_x264 = 'X264'
video_codec_divx = 'DIVX'
time_to_sleep = 180


output_directory = './testing/Rep3'
os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Naqvi2022 --output_directory {output_directory} --video_codec {video_codec_x264}')
time.sleep(time_to_sleep)
os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/vanderVoort2022 --output_directory {output_directory} --video_codec {video_codec_x264}')
time.sleep(time_to_sleep)
os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Ratnayake2023 --output_directory {output_directory} --video_codec {video_codec_divx}')
time.sleep(time_to_sleep)
os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Ratnayake2021 --output_directory {output_directory} --video_codec {video_codec_x264}')
time.sleep(time_to_sleep)
os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/PICT --output_directory {output_directory} --video_codec {video_codec_x264}')
time.sleep(time_to_sleep)
os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/JamesCook --output_directory {output_directory} --video_codec {video_codec_x264}')
time.sleep(time_to_sleep*4)

# Shoutdown 

#output_directory = './testing/Rep4'
#os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Naqvi2022 --output_directory {output_directory} --video_codec {video_codec_x264}')
#time.sleep(time_to_sleep)
#os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/vanderVoort2022 --output_directory {output_directory} --video_codec {video_codec_x264}')
#time.sleep(time_to_sleep)
#os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Ratnayake2023 --output_directory {output_directory} --video_codec {video_codec_divx}')
#time.sleep(time_to_sleep)
#os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Ratnayake2021 --output_directory {output_directory} --video_codec {video_codec_x264}')
#time.sleep(time_to_sleep)
#os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/PICT --output_directory {output_directory} --video_codec {video_codec_x264}')
#time.sleep(time_to_sleep)
#os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/JamesCook --output_directory {output_directory} --video_codec {video_codec_x264}')
#time.sleep(time_to_sleep)

# output_directory = './testing/Rep4'
# os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Naqvi2022 --output_directory {output_directory} --video_codec {video_codec_x264}')
# time.sleep(time_to_sleep)
# os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/vanderVoort2022 --output_directory {output_directory} --video_codec {video_codec_x264}')
# time.sleep(time_to_sleep)
# os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Ratnayake2023 --output_directory {output_directory} --video_codec {video_codec_divx}')
# time.sleep(time_to_sleep)
# os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Ratnayake2021 --output_directory {output_directory} --video_codec {video_codec_x264}')
# time.sleep(time_to_sleep)
# os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/PICT --output_directory {output_directory} --video_codec {video_codec_x264}')
# time.sleep(time_to_sleep)
# os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/JamesCook --output_directory {output_directory} --video_codec {video_codec_x264}')
# time.sleep(time_to_sleep)

