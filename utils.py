from ultralytics import YOLO
import cv2
import os
from datetime import datetime

def load_model(path='best.pt'):
    return YOLO(path)

def process_video(model, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        annotated_frame = results.plot()
        out.write(annotated_frame)
    cap.release()
    out.release()