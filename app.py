import streamlit as st
import cv2
from model import detect_plate
from ocr import read_text
import tempfile
import os
import csv

st.set_page_config(layout="wide")
st.title("🚗 License Plate Detector & OCR")
st.markdown("Upload a video. This app will extract and OCR all detected license plates.")

video_file = st.file_uploader("📹 Upload a video file", type=["mp4", "mov", "avi"])
output_dir = "outputs/plates"
csv_path = "outputs/plate_texts.csv"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

if video_file:
    # Save temp video file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    results = []
    plate_counter = 0
    frame_count = 0

    with st.spinner("🔍 Processing video and extracting plates..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            plates = detect_plate(frame)

            for cropped, _ in plates:
                text = read_text(cropped)
                plate_filename = f"plate_{frame_count}_{plate_counter}.png"
                plate_path = os.path.join(output_dir, plate_filename)

                cv2.imwrite(plate_path, cropped)

                results.append((plate_filename, text))
                plate_counter += 1

    cap.release()

    # Write results to CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "PlateText"])
        writer.writerows(results)

    # Show results in Streamlit
    st.success("✅ Plates extracted successfully!")
    st.download_button("📥 Download CSV", data=open(csv_path, 'rb'), file_name="plate_texts.csv")

    for filename, text in results:
        st.image(os.path.join(output_dir, filename), caption=f"Detected: {text}")
