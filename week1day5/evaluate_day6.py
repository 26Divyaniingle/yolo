
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "yolo_dataset" / "data.yaml"
BEST_MODEL = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "day5_yolov8n"
    / "weights"
    / "best.pt"
)

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

if not BEST_MODEL.exists():
    raise FileNotFoundError(f"Trained model not found: {BEST_MODEL}")

# Load the best trained checkpoint
model = YOLO(str(BEST_MODEL))

# Evaluate on the held-out test split
metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=8,
    plots=True,
    project=str(PROJECT_DIR / "runs" / "detect"),
    name="day6_test_evaluation",
    exist_ok=True,
)

# Print overall metrics
print("\n===== DAY 6 TEST RESULTS =====")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")
print(f"mAP50:       {metrics.box.map50:.4f}")
print(f"mAP50-95:    {metrics.box.map:.4f}")

# Print per-class AP
print("\n===== PER-CLASS RESULTS =====")
for class_id, ap50 in enumerate(metrics.box.ap50):
    class_name = model.names[class_id]
    ap50_95 = metrics.box.ap[class_id]
    print(
        f"{class_name}: "
        f"mAP50={ap50:.4f}, "
        f"mAP50-95={ap50_95:.4f}"
    )

print("\nEvaluation files saved in:")
print(PROJECT_DIR / "runs" / "detect" / "day6_test_evaluation")