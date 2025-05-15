from ultralytics import YOLO
import cv2

def load_model(path='best.pt'):
    return YOLO(path)

def process_video(model, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated_frame = results.plot()
        out.write(annotated_frame)

    cap.release()
    out.release()