import csv
import json
from pathlib import Path
import cv2
from ultralytics import YOLO

# -------------------------------------------------------------
# 1. Setup paths
# -------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = SCRIPT_DIR / "images"
if not IMAGES_DIR.exists():
    IMAGES_DIR = Path("images")

RUNS_DIR = SCRIPT_DIR / "runs"
ANNOTATED_DIR = RUNS_DIR / "annotated"
CROPS_DIR = RUNS_DIR / "crops"
CSV_PATH = SCRIPT_DIR / "predictions.csv"
RUNS_CSV_PATH = RUNS_DIR / "predictions.csv"
JSON_PATH = RUNS_DIR / "predictions.json"

# Create output directories
RUNS_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 2. Load model and run prediction
# -------------------------------------------------------------
model = YOLO("yolov8n.pt")

results = model.predict(
    source=str(IMAGES_DIR),
    conf=0.25,
    imgsz=640,
    device="cpu"
)

# List to accumulate detection rows for CSV and JSON export
csv_rows = []
json_records = []

# -------------------------------------------------------------
# 3. Process detections: boxes.xyxy, conf, cls, names, plot, crops
# -------------------------------------------------------------
for result in results:
    image_name = Path(result.path).name
    image_stem = Path(result.path).stem
    boxes = result.boxes

    print(f"\nImage: {result.path}")

    if boxes is None or len(boxes) == 0:
        print("No detections")
        continue

    image_detections = []

    for i in range(len(boxes)):
        # 1. Bounding box coordinates (boxes.xyxy)
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

        # 2. Confidence score (boxes.conf)
        confidence = float(boxes.conf[i])

        # 3. Class ID (boxes.cls)
        class_id = int(boxes.cls[i])

        # 4. Class name via names dict (result.names)
        class_name = result.names[class_id]

        print(
            f"Class: {class_name}, "
            f"Confidence: {confidence:.2f}, "
            f"Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )

        # Append row for CSV export
        csv_rows.append({
            "image_name": image_name,
            "class": class_name,
            "confidence": round(confidence, 4),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2)
        })

        image_detections.append({
            "class": class_name,
            "confidence": round(confidence, 4),
            "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

    json_records.append({
        "image_name": image_name,
        "detections": image_detections
    })

    # 5. Plotting with results.plot() and saving annotated image
    annotated_img = result.plot()
    annotated_output_path = ANNOTATED_DIR / f"{image_stem}_annotated.jpg"
    cv2.imwrite(str(annotated_output_path), annotated_img)

    # 6. Saving crops (saving crops per detected object)
    result.save_crop(save_dir=CROPS_DIR, file_name=Path(image_name))

# -------------------------------------------------------------
# 4. Export predictions to CSV: image_name, class, confidence, x1, y1, x2, y2
# -------------------------------------------------------------
csv_fieldnames = ["image_name", "class", "confidence", "x1", "y1", "x2", "y2"]

for target_csv in [CSV_PATH, RUNS_CSV_PATH]:
    with open(target_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

print(f"\nSaved CSV predictions to: {CSV_PATH}")
print(f"Saved CSV copy to: {RUNS_CSV_PATH}")

# -------------------------------------------------------------
# 5. Export predictions to JSON
# -------------------------------------------------------------
with open(JSON_PATH, mode="w", encoding="utf-8") as f:
    json.dump(json_records, f, indent=4)

print(f"Saved JSON predictions to: {JSON_PATH}")
print(f"Annotated images saved to: {ANNOTATED_DIR}")
print(f"Object crops saved to: {CROPS_DIR}")
print(f"Total detections saved: {len(csv_rows)}")