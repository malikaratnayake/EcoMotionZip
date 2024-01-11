#!/bin/bash

# Define the CSV file path
csv_file="/home/pi-cam7/EcoMotionZip/cpu_temps.csv"

# Get the current timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Get the CPU temperature
cpu_temp=$(vcgencmd measure_temp | cut -d'=' -f2)

# Ensure the CSV file exists
if [[ ! -f "$csv_file" ]]; then
  echo "Creating new CSV file: $csv_file"
  echo "Timestamp,CPU Temperature (C)" >> "$csv_file"
fi

# Append temperature data to CSV file
echo "$timestamp,$cpu_temp" >> "$csv_file"

# Write a cron job to run this script every 5 seconds
# crontab -e
# */5 * * * * /home/pi/pi_cpu_record.sh

