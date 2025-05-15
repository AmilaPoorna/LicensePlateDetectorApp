import streamlit as st
from utils import load_model, process_video
from datetime import datetime
import tempfile
import os

st.set_page_config(page_title="License Plate Detector App", layout="centered")
st.title("License Plate Detector")

uploaded_file = st.file_uploader("Upload a video:", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    model = load_model("best.pt")

    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = tmp_output.name

    stframe = st.empty()
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def show_frame(frame, current, total):
        stframe.image(frame, channels="BGR", caption=f"Frame {current} of {total}")
        progress_bar.progress(current / total)
        progress_text.text(f"Processing: {current}/{total} frames")

    try:
        with st.spinner("🔍 Processing video..."):
            process_video(model, input_path, output_path, show_frame)

        st.success("✅ Video processed successfully!")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("📥 Download Processed Video", f, file_name="processed_video.mp4")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")