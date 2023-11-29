import os

output_directory = './testing'
video_codec = 'XVID'

command = f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Ratnayake2023 --output_directory {output_directory} --video_codec {video_codec}'
os.system(command)


output_directory = './testing'
video_codec = 'X264'

command = f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Ratnayake2021 --output_directory {output_directory} --video_codec {video_codec}'
os.system(command)

output_directory = './testing'
video_codec = 'X264'

command = f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/Naqvi2022 --output_directory {output_directory} --video_codec {video_codec}'
os.system(command)

output_directory = './testing'
video_codec = 'X264'

command = f'python EcoMotionZip/app.py --video_source /home/pi-cam7/EcoMotionZip/testing/test_videos/vanderVoort2022 --output_directory {output_directory} --video_codec {video_codec}'
os.system(command)

