# 🚘 License Plate Detector App

This is a simple Streamlit web application that allows users to upload a video and detect license plates using a trained YOLOv8 model. The app annotates videos highlighting the detected license plates, which can be downloaded directly from the app too.

## 📸 Demo

![Demo]([demo.gif](https://github.com/user-attachments/assets/cc537905-6546-4262-905b-e8f773fbbf89))  
*Replace with actual GIF or image of your app.*

---

## 🔧 Features

- 📁 Upload videos in `.mp4`, `.avi`, or `.mov` format
- 🧠 Automatically detect license plates using a custom-trained YOLOv8 model (`best.pt`)
- 🎞 Annotates license plates frame-by-frame in real-time
- 💾 Allows downloading of the processed video
- 📊 Displays progress while processing the video

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/license-plate-streamlit-app.git
   cd license-plate-streamlit-app
