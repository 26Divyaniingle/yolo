
from pathlib import Path
from ultralytics import YOLO

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "yolo_dataset" / "data.yaml"

# 2. Check dataset configuration
if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

# 3. Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# 4. Train with modified augmentation
results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    optimizer="auto",
    pretrained=True,

    # Augmentation experiment
    mosaic=0.8,
    mixup=0.1,

    # Keep the remaining settings comparable
    project=str(BASE_DIR / "runs" / "detect"),
    name="day6_exp1_aug",
    exist_ok=True,
    plots=True,
    save=True,
    workers=2,
    device="cpu"
)

print("\nExperiment 1 training finished.")
print("Best weights:")
print(BASE_DIR / "runs" / "detect" /
      "day6_exp1_aug" / "weights" / "best.pt")