import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import os

st.title("License Plate Detector")

# Load model once
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Prepare output path
    output_path = "annotated_output.mp4"

    # Read video
    cap = cv2.VideoCapture(video_path)

    # Video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()  # placeholder for video frames

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame)
        annotated_frame = results[0].plot()  # returns annotated frame
        
        # Write frame to output video
        out.write(annotated_frame)

        # Convert BGR to RGB for Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame_rgb, channels="RGB")

        # Update progress bar
        progress_bar.progress((i + 1) / frame_count)
    
    cap.release()
    out.release()

    st.success("Video processing complete!")

    # Provide download link
    with open(output_path, "rb") as f:
        st.download_button(
            label="Download Annotated Video",
            data=f,
            file_name="annotated_video.mp4",
            mime="video/mp4"
        )