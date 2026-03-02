#!/bin/bash
set -e  # exit on error

# Where am I?
EXERCISE_DIR=$(realpath ${BASH_SOURCE[0]} | xargs dirname)

# Go to CameraHMR folder
cd ${EXERCISE_DIR}/CameraHMR


# Install torch (As of right now, the pool computers support CUDA 12.5)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Install requirements
pip install -r requirements.txt


# Create temporary folder for user
mkdir -p /tmp/${USER}/CameraHMR_data
ln -s /tmp/${USER}/CameraHMR_data data

# Download the trained model
bash scripts/fetch_demo_data.sh
