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

if 'process' not in st.session_state:
    st.session_state.process = False
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'last_video' not in st.session_state:
    st.session_state.last_video = None

uploaded_video = st.file_uploader("Upload a Video:", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    if uploaded_video.name != st.session_state.last_video:
        st.session_state.process = False
        st.session_state.output_path = None
        st.session_state.last_video = uploaded_video.name

    if not st.session_state.process:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        progress_bar = st.progress(0)

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            processed_frame = results[0].plot()
            out.write(processed_frame)

            stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            progress_bar.progress((i + 1) / frame_count)

        cap.release()
        out.release()

        st.success("Done!")

        st.session_state.process = True
        st.session_state.output_path = output_path

if st.session_state.process and st.session_state.output_path:
    with open(st.session_state.output_path, "rb") as f:
        st.download_button(
            label="Download the Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )