
from pathlib import Path
import csv

import cv2
from ultralytics import YOLO


# ---------- Paths and settings ----------
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "yolo_dataset"
IMAGE_DIR = DATASET / "images" / "test"
LABEL_DIR = DATASET / "labels" / "test"

MODEL_PATH = (
    ROOT / "runs" / "detect" / "day5_yolov8n"
    / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "runs" / "detect" / "day6_worst_predictions"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"

CONFIDENCE = 0.25
IOU_MATCH_THRESHOLD = 0.50
TOP_N = 10

IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_iou(box_a, box_b):
    """Calculate IoU between two xyxy bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def read_ground_truth(label_path, width, height):
    """Read YOLO labels and convert normalized xywh to pixel xyxy."""
    ground_truth = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label file: {label_path}"
        )

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        values = line.split()
        class_id = int(values[0])
        x_center, y_center, box_width, box_height = map(
            float, values[1:5]
        )

        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height

        ground_truth.append({
            "class_id": class_id,
            "box": [x1, y1, x2, y2],
        })

    return ground_truth


def annotate_image(image, ground_truth, predictions, model_names):
    """Draw ground truth in green and predictions in red."""
    canvas = image.copy()

    for gt in ground_truth:
        x1, y1, x2, y2 = map(int, gt["box"])
        class_name = model_names[gt["class_id"]]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            canvas,
            f"GT: {class_name}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 150, 0),
            2,
        )

    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred["box"])
        class_name = model_names[pred["class_id"]]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"Pred: {class_name} {pred['confidence']:.2f}",
            (x1, min(canvas.shape[0] - 5, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    return canvas


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise FileNotFoundError("Test images or labels folder is missing.")

    model = YOLO(str(MODEL_PATH))
    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        raise RuntimeError(f"No test images found in {IMAGE_DIR}")

    ranked = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        height, width = image.shape[:2]
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        gt = read_ground_truth(label_path, width, height)

        result = model.predict(
            source=image,
            conf=CONFIDENCE,
            imgsz=640,
            verbose=False,
        )[0]

        predictions = []
        if result.boxes is not None:
            for box in result.boxes:
                predictions.append({
                    "class_id": int(box.cls.item()),
                    "confidence": float(box.conf.item()),
                    "box": box.xyxy[0].cpu().tolist(),
                })

        used_predictions = set()
        false_negatives = 0
        false_positives = 0
        class_confusions = 0

        # Match each ground-truth object to the best remaining prediction.
        for target in gt:
            best_index = None
            best_iou = 0.0

            for index, pred in enumerate(predictions):
                if index in used_predictions:
                    continue

                overlap = calculate_iou(target["box"], pred["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index

            if (
                best_index is not None
                and best_iou >= IOU_MATCH_THRESHOLD
            ):
                used_predictions.add(best_index)

                if (
                    predictions[best_index]["class_id"]
                    != target["class_id"]
                ):
                    class_confusions += 1
            else:
                false_negatives += 1

        false_positives = len(predictions) - len(used_predictions)
        error_count = (
            false_negatives + false_positives + class_confusions
        )

        if error_count > 0:
            ranked.append({
                "image": image_path,
                "gt": gt,
                "predictions": predictions,
                "false_negatives": false_negatives,
                "false_positives": false_positives,
                "class_confusions": class_confusions,
                "error_count": error_count,
            })

    # Rank images by the number of detected error cases.
    ranked.sort(
        key=lambda item: (
            item["error_count"],
            item["class_confusions"],
            item["false_negatives"],
            item["false_positives"],
        ),
        reverse=True,
    )

    report_path = OUTPUT_DIR / "top10_errors.csv"

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "image",
                "error_count",
                "false_negatives",
                "false_positives",
                "class_confusions",
                "root_cause_after_visual_inspection",
            ],
        )
        writer.writeheader()

        for rank, item in enumerate(ranked[:TOP_N], start=1):
            image = cv2.imread(str(item["image"]))
            annotated = annotate_image(
                image,
                item["gt"],
                item["predictions"],
                model.names,
            )

            saved_image = IMAGE_OUTPUT_DIR / (
                f"{rank:02d}_{item['image'].stem}.jpg"
            )
            cv2.imwrite(str(saved_image), annotated)

            writer.writerow({
                "rank": rank,
                "image": item["image"].name,
                "error_count": item["error_count"],
                "false_negatives": item["false_negatives"],
                "false_positives": item["false_positives"],
                "class_confusions": item["class_confusions"],
                "root_cause_after_visual_inspection":
                    "Inspect: bad label / small object / occlusion / "
                    "class confusion / blur / other",
            })

            print(
                f"{rank:02d}. {item['image'].name}: "
                f"errors={item['error_count']}, "
                f"FN={item['false_negatives']}, "
                f"FP={item['false_positives']}, "
                f"class confusion={item['class_confusions']}"
            )

    print(f"\nTop-error CSV: {report_path}")
    print(f"Annotated images: {IMAGE_OUTPUT_DIR}")
    print("\nInspect the saved images before assigning root causes.")


if __name__ == "__main__":
    main()