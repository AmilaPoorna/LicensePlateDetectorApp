from ultralytics import YOLO
import cv2

def load_model(path='best.pt'):
    return YOLO(path)

def process_video(model, input_path, output_path, progress_callback=None):
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        annotated_frame = results.plot()
        out.write(annotated_frame)

        frame_count += 1
        if progress_callback:
            progress_callback(annotated_frame, frame_count, total_frames)

    cap.release()
    out.release()