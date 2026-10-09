
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "yolo_dataset" / "data.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

model = YOLO("yolov8n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=20,
    imgsz=768,          # Experiment 2: higher resolution
    batch=8,
    patience=10,
    optimizer="auto",
    pretrained=True,

    # Keep Experiment 1 augmentation settings unchanged
    mosaic=0.8,
    mixup=0.1,

    project=str(BASE_DIR / "runs" / "detect"),
    name="day6_exp2_imgsz768",
    exist_ok=True,
    plots=True,
    save=True,
    workers=2,
    device="cpu"
)

print("\nExperiment 2 training finished.")
print(
    BASE_DIR / "runs" / "detect" /
    "day6_exp2_imgsz768" / "weights" / "best.pt"
)