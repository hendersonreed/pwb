#!/bin/env python3

import time
import pyautogui as pg

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2 as cv

model_path = '/home/reed/recurse/pwb-projector-whiteboard/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


WIDTH, HEIGHT = pg.size()


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
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
        print(f"would move to {x},{y}")
        # we don't want to run `moveTo()` every time this function runs, rather we need to drop some percentage of them... could do it based on timestamp_ms?
        pg.moveTo(x, y)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
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
