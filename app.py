import streamlit as st
from pathlib import Path
import tempfile
import os
from utils import load_model, process_video

st.set_page_config(page_title="License Plate Detector App", layout="wide")

st.title("License Plate Detector")

uploaded_file = st.file_uploader("Upload a video:", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_video_path = Path(tfile.name)

    st.video(str(input_video_path))

    if st.button("Start Processing"):

        model = load_model("best.pt")

        output_video_path = input_video_path.parent / f"output_{input_video_path.name}"

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(progress):
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing: {int(progress*100)}%")

        process_video(input_video_path, output_video_path, model, progress_callback)

        st.success("Done!")
        st.video(str(output_video_path))

        with open(output_video_path, "rb") as f:
            video_bytes = f.read()
            st.download_button(
                label="Download the Processed Video",
                data=video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
