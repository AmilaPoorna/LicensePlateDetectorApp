import streamlit as st
from utils import load_model, process_video
from datetime import datetime
import os

st.set_page_config(page_title="License Plate Detector", layout="centered")
st.title("License Plate Detector App")

os.makedirs("outputs", exist_ok=True)

uploaded_file = st.file_uploader("Upload a video:", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    model = load_model("best.pt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/processed_{timestamp}.mp4"

    try:
        with st.spinner("🔍 Processing the video..."):
            process_video(model, "temp_video.mp4", output_path)

        st.success("✅ The video was processed successfully.")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("Download the Processed Video", f, file_name="processed_video.mp4")

    except Exception as e:
        st.error(f"❌ An error has occured: {e}")