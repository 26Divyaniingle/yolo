from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Three confidence thresholds
confidence_values = [0.25, 0.50, 0.75]

# Run prediction for each confidence value
for conf in confidence_values:

    print("\n" + "=" * 50)
    print(f"Running YOLO with confidence = {conf}")
    print("=" * 50)

    results = model.predict(
        source="images/",
        conf=conf,
        iou=0.5,
        imgsz=640,
        device="cpu",
        save=True,
        project=r"C:\Users\Desktop\computervisionintern\week1day4\runs",
        name=f"conf_{conf}",
        exist_ok=True
    )

    total_detections = 0

    for result in results:
        if result.boxes is not None:
            total_detections += len(result.boxes)

    print(f"Total detections at conf={conf}: {total_detections}")