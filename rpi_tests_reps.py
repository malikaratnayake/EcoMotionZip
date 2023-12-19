import os
import time

#output_directory = './testing/Rep1'
video_codec_x264 = 'X264'
video_codec_divx = 'DIVX'
time_to_sleep = 120
dataset = 'PICT'


rep_1 = './testing/Rep1'
rep_2 = './testing/Rep2'
rep_3 = './testing/Rep3'

# Naqvi2022
time.sleep(30)
os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/{dataset} --output_directory {rep_1} --video_codec {video_codec_x264}')
time.sleep(time_to_sleep)
os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/{dataset} --output_directory {rep_2} --video_codec {video_codec_x264}')
time.sleep(time_to_sleep)
os.system(f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/{dataset} --output_directory {rep_3} --video_codec {video_codec_x264}')
time.sleep(time_to_sleep)

# # vanderVoort2022



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
# time.sleep(time_to_sleep*4)

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

