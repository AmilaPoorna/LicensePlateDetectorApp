import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import torch
import os

torch.classes.__path__ = []

st.title("License Plate Detector")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    output_path = "processed_output.mp4"

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty() 

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        processed_frame = results[0].plot()
        
        out.write(processed_frame)

        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(processed_frame_rgb, channels="RGB")

        progress_bar.progress((i + 1) / frame_count)
    
    cap.release()
    out.release()

    st.success("Done!")

    with open(output_path, "rb") as f:
        st.download_button(
            label="Download the Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )