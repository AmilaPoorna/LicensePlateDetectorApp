from ultralytics import YOLO
import cv2

def load_model(path='best.pt'):
    return YOLO(path)

def process_video(model, input_path, output_path, progress_callback=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("Failed to write output video.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated = results.plot()
        out.write(annotated)
        count += 1

        if progress_callback:
            progress_callback(annotated, count, total)

    cap.release()
    out.release()
