#!/bin/env python3

import time
import pyautogui as pg

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2 as cv


##################
# CONFIG OPTIONS #
##################

MODEL_PATH = '/home/reed/recurse/pwb-projector-whiteboard/hand_landmarker.task'
WIDTH, HEIGHT = pg.size()


###############
# ACTUAL CODE #
###############

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

last_execution_time = time.time()
time_threshold = 10


def track_finger_with_mouse(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """
    it's in here that we need to evaluate the result
    1. move the mouse to the same x-y of the index finger tip using pyautogui
        # x = int(result['hand_landmarks'][0][8].x)
        # y = int(result['hand_landmarks'][0][8].y)
        # pg.moveTo(x*WIDTH,y*HEIGHT)
    2. check the distance between middle-finger tip and thumb-tip.
        - if it's below the proper threshold, we can execute a click or alert or something.
    """
    if len(result.hand_landmarks) > 0:
        x = int(result.hand_landmarks[0][8].x * WIDTH)
        y = int(result.hand_landmarks[0][8].y * HEIGHT)

        current_time = time.time()
        if current_time - last_execution_time >= time_threshold:
            print(f"moving to {x},{y}")
            pg.moveTo(x, y)
            last_execution_time = current_time


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=track_finger_with_mouse)
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        timestamp = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)

        cv.imshow('window', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
