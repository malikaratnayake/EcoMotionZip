import os
import subprocess
import csv
from datetime import datetime

client_directory = "./testing/test_videos"
server_directory = "/home/pi-server/EcoMotionZip/Originals"
server_address = "10.42.0.1"
csv_file = "transfer_times.csv"

# Create the CSV file and write the header
with open(csv_file, "w") as file:
    writer = csv.writer(file)
    writer.writerow(["File", "Transfer Time"])

# Iterate through all subdirectories and files in the client directory
for root, dirs, files in os.walk(client_directory):
    for file in files:
        # Get the full path of the fileMo
        file_path = os.path.join(root, file)

        # Get the relative path of the file (without the client directory)
        relative_path = os.path.relpath(file_path, client_directory)

        # Construct the destination path on the server
        destination_path = os.path.join(server_directory, relative_path)
        folder_name = os.path.dirname(relative_path)

        destination_subdir = os.path.join(server_directory, folder_name)
        # print(folder_name, destination_path, relative_path)

        # Create the destination directory
        os.system("ssh {} mkdir -p {}".format("pi-server@10.42.0.1", destination_subdir))

        # Run the scp command to transfer the file to the server
        start_time = datetime.now()
        subprocess.run(["scp", file_path, f"pi-server@{server_address}:{destination_path}"])
        end_time = datetime.now()

        # Calculate the transfer time
        transfer_time = end_time - start_time

        # Append the transfer time to the CSV file
        with open(csv_file, "a") as file:
            writer = csv.writer(file)
            writer.writerow([relative_path, transfer_time.total_seconds()])
