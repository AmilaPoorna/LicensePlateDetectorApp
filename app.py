import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("model/best.pt")

st.title("License Plate Detector")

uploaded_video = st.file_uploader("Upload a video:", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    input_path = tfile.name

    st.video(input_path)
    st.write("Processing the video...")

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join("output", os.path.basename(input_path))
    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    progress = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

        if i % 5 == 0:
            st.image(frame, channels="BGR", caption=f"Frame {i}", use_container_width=True)

        progress.progress((i+1) / frame_count)

    cap.release()
    out.release()
    st.success("Done!")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button("Download the Processed Video", f, file_name="processed_video.mp4")