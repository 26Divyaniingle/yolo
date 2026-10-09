
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "runs" / "detect" / "day6_exp1_aug"
    / "weights" / "best.pt"
)

DATA_YAML = BASE_DIR / "yolo_dataset" / "data.yaml"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=640,
    batch=8,
    plots=True,
    project=str(BASE_DIR / "runs" / "detect"),
    name="day6_exp1_validation",
    exist_ok=True,
    device="cpu"
)

print("\n--- EXPERIMENT 1 VALIDATION RESULTS ---")
print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")
print(f"mAP50:      {metrics.box.map50:.4f}")
print(f"mAP50-95:   {metrics.box.map:.4f}")