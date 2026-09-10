import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import pathlib

#==============================================================================
#Open Camera object
cap = cv2.VideoCapture(0)
model_path = pathlib.Path("../runs/pose/train-3/weights")
model = YOLO(model_path / "best.pt")

while cv2.waitKey(2) != ord("x"):
    data, image = cap.read()

    #Flit the image to selfie view of the webcam
    image = cv2.flip(image, 1)

    result = model(image)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Tracker', result[0].plot())

cv2.waitKey(5000)
cap.release()