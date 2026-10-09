from pathlib import Path
import yaml
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
YAML_PATH = PROJECT_DIR / "yolo_dataset" / "data.yaml"

with open(YAML_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

dataset_root = Path(config["path"])

if not dataset_root.is_absolute():
    dataset_root = (YAML_PATH.parent / dataset_root).resolve()

empty_label_names = {
    "BikesHelmets279",
    "BikesHelmets35",
    "BikesHelmets441",
    "BikesHelmets459",
    "BikesHelmets326",
    "BikesHelmets735",
    "BikesHelmets764",
    "BikesHelmets103",
}

image_extensions = [
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
]

for split in ["train", "val", "test"]:

    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    print(f"\nInspecting {split.upper()} split")

    for label_path in label_dir.glob("*.txt"):

        if label_path.stem not in empty_label_names:
            continue

        if label_path.read_text(encoding="utf-8").strip():
            continue

        image_path = None

        for extension in image_extensions:
            candidate = image_dir / f"{label_path.stem}{extension}"

            if candidate.exists():
                image_path = candidate
                break

        print(f"\nLabel: {label_path.name}")

        if image_path is None:
            print("Corresponding image not found.")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read image:", image_path)
            continue

        print("Image:", image_path.name)
        print("Image size:", image.shape[1], "x", image.shape[0])

        # Resize large images for easier viewing.
        height, width = image.shape[:2]
        scale = min(1000 / width, 700 / height, 1.0)

        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale))
            )

        cv2.imshow(f"{split} - {image_path.name}", image)

        print("Close this image window to continue.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("\nInspection completed.")
