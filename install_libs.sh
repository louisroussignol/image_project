#!/bin/bash

echo "Installing Python libraries..."

# Upgrade pip
python3 -m pip install --upgrade pip

# Install the listed libraries
pip3 install numpy
pip3 install opencv-python
pip3 install mediapipe
pip3 install pyautogui

echo "Installation complete!"
