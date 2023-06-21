#!/bin/bash

# Directory location
dir_path="./orchestrate_configs/"

# Iterate over each file in the directory
for file in $dir_path*
do
  # Ensure only files are processed
  if [ -f "$file" ]; then
    # Call the python script with the file name as argument
    echo "Processing $file"
    python3 train.py -c $file
    killall Google\ Chrome
    sleep 10
  fi
done
