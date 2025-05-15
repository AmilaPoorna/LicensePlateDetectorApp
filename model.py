from ultralytics import YOLO

model = YOLO("best.pt")

def detect_plate(frame):
    results = model(frame, imgsz=640)[0]
    plates = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cropped = frame[y1:y2, x1:x2]
        plates.append((cropped, (x1, y1, x2, y2)))
    return plates