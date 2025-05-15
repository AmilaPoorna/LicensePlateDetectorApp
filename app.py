import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("model/best.pt")

st.title("License Plate Detector")

uploaded_video = st.file_uploader("Upload a video:", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    input_path = tfile.name

    # Display original uploaded video
    st.video(input_path)
    st.write("Processing the video...")

    # Read video
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output video
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", os.path.basename(input_path))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Streamlit UI elements
    progress = st.progress(0)
    frame_placeholder = st.empty()

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO model prediction
        results = model(frame)[0]

        # Draw bounding boxes
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write to output video
        out.write(frame)

        # Display current frame
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

        # Update progress bar
        progress.progress((i + 1) / frame_count)

    cap.release()
    out.release()

    st.success("Processing complete!")
    st.video(output_path)

    # Download button
    with open(output_path, "rb") as f:
        st.download_button("Download the Processed Video", f, file_name="processed_video.mp4")