import streamlit as st
import cv2
from model import detect_plate
from ocr import read_text
import tempfile

st.set_page_config(layout="wide")
st.title("🚗 License Plate Recoginizer")
st.markdown("Upload a video to recognize and annotate vehicle license plates.")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
output_path = "outputs/annotated_output.mp4"

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()

    with st.spinner("🔍 Processing the video..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            plates = detect_plate(frame)
            for cropped, (x1, y1, x2, y2) in plates:
                text = read_text(cropped)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            out.write(frame)

            preview = cv2.resize(frame, (640, 360))
            stframe.image(preview, channels="BGR")

    cap.release()
    out.release()

    st.success("✅ Processing complete!")
    st.video(output_path)