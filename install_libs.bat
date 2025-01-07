@echo off
echo Installing Python libraries...

REM Ensure Python is in PATH or use the full path to python.exe
python -m pip install --upgrade pip

REM Install the listed libraries
pip install numpy
pip install opencv-python
pip install mediapipe
pip install pyautogui

echo Installation complete!
pause
