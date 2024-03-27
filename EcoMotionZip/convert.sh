# Wite a bash script to convert the video files in the test_videos folder and deinterlace them. Use the following code snippet to get started:
# ffmpeg -i input.vob -vf yadif -c:v libx264 -preset slow -crf 19 -c:a aac -b:a 256k output.mp4

# The script should:
# 1. Loop through all the files in the test_videos folder
# 2. Check if the file is a video file
# 3. If it is a video file, convert it to mp4 and deinterlace it
# 4. Save the converted file in the test_videos_output folder




#!/bin/bash

input_folder="/Users/mrat0010/Documents/Bluebanded_bee-James/1st_november_apiary"
output_folder="/Users/mrat0010/Documents/Bluebanded_bee-James/1st_Nov_apiary-preprocessed"

for file in "$input_folder"/*; do
    if [[ -f "$file" && ( "$file" == *.mp4 || "$file" == *.MTS ) ]]; then
        filename=$(basename "$file")
        output_file="$output_folder/${filename%.*}.mp4"
        ffmpeg -i "$file" -vf yadif -c:v libx264 -preset slow -crf 19 -c:a aac -b:a 256k "$output_file"
    fi
done