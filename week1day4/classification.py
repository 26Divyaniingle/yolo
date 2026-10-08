from ultralytics import YOLO
from pathlib import Path

# Base directory of this script
BASE_DIR = Path(__file__).resolve().parent

# Pretrained classification model
model_path = str(BASE_DIR / "yolo11n-cls.pt") if (BASE_DIR / "yolo11n-cls.pt").exists() else "yolo11n-cls.pt"
model = YOLO(model_path)

image_path = str(BASE_DIR / "images" / "image4.jpg")

results = model(image_path, verbose=False)

result = results[0]

top_class_id = result.probs.top1

confidence = float(result.probs.top1conf)

class_name = result.names[top_class_id]

print("Predicted class:", class_name)

print("Confidence:", round(confidence, 4))