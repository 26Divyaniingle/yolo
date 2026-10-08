from pathlib import Path
import yaml


# --------------------------------------------------
# 1. Locate dataset
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

YAML_PATH = PROJECT_DIR / "yolo_dataset" / "data.yaml"

with open(YAML_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# --------------------------------------------------
# 2. Dataset configuration
# --------------------------------------------------

dataset_root = Path(config["path"])

if not dataset_root.is_absolute():
    dataset_root = YAML_PATH.parent / dataset_root

dataset_root = dataset_root.resolve()


# --------------------------------------------------
# 3. Files to investigate
# --------------------------------------------------

invalid_files = {
    "BikesHelmets579.txt",
    "BikesHelmets649.txt",
    "BikesHelmets680.txt",
    "BikesHelmets256.txt",
}


# --------------------------------------------------
# 4. Inspect bounding boxes
# --------------------------------------------------

for split in ["train", "val", "test"]:

    label_dir = dataset_root / "labels" / split

    image_dir = dataset_root / "images" / split

    if not label_dir.exists():
        print("Label directory not found:", label_dir)
        continue

    for label_path in label_dir.glob("*.txt"):

        if label_path.name not in invalid_files:
            continue

        print("\n" + "=" * 50)
        print("Split:", split)
        print("Label:", label_path.name)

        # Find corresponding image
        matching_images = [
            path
            for path in image_dir.glob("*")
            if path.stem == label_path.stem
        ]

        if not matching_images:
            print("Corresponding image not found.")
            continue

        image_path = matching_images[0]

        print("Image:", image_path.name)

        with open(label_path, "r", encoding="utf-8") as file:

            lines = file.readlines()

        for line_number, line in enumerate(lines, start=1):

            values = line.split()

            if len(values) != 5:
                print(
                    f"Line {line_number}: "
                    "Incorrect number of values"
                )
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])

            width = float(values[3])
            height = float(values[4])

            # Calculate normalized box boundaries
            x1 = x_center - width / 2
            y1 = y_center - height / 2

            x2 = x_center + width / 2
            y2 = y_center + height / 2

            print(f"\nLine: {line_number}")
            print("Class ID:", class_id)

            print(
                f"Left={x1:.6f}, "
                f"Top={y1:.6f}, "
                f"Right={x2:.6f}, "
                f"Bottom={y2:.6f}"
            )

            print(
                f"Center=({x_center:.6f}, {y_center:.6f}), "
                f"Size=({width:.6f}, {height:.6f})"
            )

            if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                print("STATUS: Bounding box extends outside image.")
            else:
                print("STATUS: Bounding box is inside image.")

print("\nInspection completed.")