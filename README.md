# Driver-Attention-Monitoring-System

This is a student project developed during my undergraduate studies together with two colleagues. It detects driver drowsiness and distraction, triggering alerts if the eyes are closed for more than 5 seconds or if the driver looks away from the road. This version was later upgraded with STM32 microcontrollers for better real-time performance and hardware integration.

## Requirements
Python 3.x

Install the needed libraries:
pip install opencv-python pillow numpy face_recognition dlib scipy pygame imutils

## How to run
Run:
python run.py

## Description
After running the project, a simple GUI will open.  
You can choose an image or video, and the system will detect the eyes and head direction.  
If the driver is not paying attention or keeps their eyes closed for too long, an alarm is triggered.
