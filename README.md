# 🚘 License Plate Detector App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://licenseplatedetectorapp-a3h3kbmn8xmn7ds4a4yu2s.streamlit.app/)

This is a simple Streamlit web application that allows users to upload a video and detect license plates using a trained YOLOv8 model. The app annotates the video by highlighting the detected license plates, which can then be downloaded directly from the app.

## Demo

![Demo](https://github.com/user-attachments/assets/cc537905-6546-4262-905b-e8f773fbbf89)  
*GIF of an annotated video.*

## Features

- Allows users to upload videos in `.MP4`, `.AVI`, `.MOV` or `.MPEG4` format.
- Detects license plates using a trained `YOLOv8n` model on **License-Plate-Recognition-11** dataset from Roboflow Universe.
- Displays frame-by-frame annotations of detected license plates
- Displays progress while processing the video.
- Allows users to download processed video.
