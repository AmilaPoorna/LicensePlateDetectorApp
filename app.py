import streamlit as st
import cv2
from model import detect_plates
from ocr import read_plate_text
import tempfile

st.title("🚗 License Plate Recognizer")

video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
output_path = "outputs/annotated_output.mp4"

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with st.spinner("Processing video..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            plates = detect_plates(frame)
            for cropped, (x1, y1, x2, y2) in plates:
                text = read_plate_text(cropped)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            out.write(frame)

    cap.release()
    out.release()

    st.success("✅ Processing complete!")
    st.video(output_path)