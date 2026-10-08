from pathlib import Path
import yaml

# Project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset configuration
YAML_PATH = PROJECT_DIR / "yolo_dataset" / "data.yaml"

with open(YAML_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

dataset_root = Path(config["path"])

if not dataset_root.is_absolute():
    dataset_root = (YAML_PATH.parent / dataset_root).resolve()

# Only correct the four files identified by the validator
invalid_files = {
    "BikesHelmets579.txt",
    "BikesHelmets649.txt",
    "BikesHelmets680.txt",
    "BikesHelmets256.txt",
}

fixed_count = 0

for split in ["train", "val", "test"]:

    label_dir = dataset_root / "labels" / split

    if not label_dir.exists():
        print(f"Label directory not found: {label_dir}")
        continue

    for label_path in label_dir.glob("*.txt"):

        if label_path.name not in invalid_files:
            continue

        original_lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        corrected_lines = []

        for line in original_lines:

            values = line.split()

            if len(values) != 5:
                corrected_lines.append(line)
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            # Convert center coordinates into box boundaries
            left = x_center - width / 2
            top = y_center - height / 2
            right = x_center + width / 2
            bottom = y_center + height / 2

            # Correct only tiny boundary overflows.
            # Do not silently repair substantial errors.
            tolerance = 0.00001

            if (
                left < -tolerance
                or top < -tolerance
                or right > 1 + tolerance
                or bottom > 1 + tolerance
            ):
                raise ValueError(
                    f"Large boundary error in {label_path.name}: {line}"
                )

            left = max(0.0, left)
            top = max(0.0, top)
            right = min(1.0, right)
            bottom = min(1.0, bottom)

            # Recalculate YOLO center and dimensions
            new_width = right - left
            new_height = bottom - top

            if new_width <= 0 or new_height <= 0:
                raise ValueError(
                    f"Invalid box dimensions in {label_path.name}"
                )

            new_x_center = (left + right) / 2
            new_y_center = (top + bottom) / 2

            corrected_line = (
                f"{class_id} "
                f"{new_x_center:.8f} "
                f"{new_y_center:.8f} "
                f"{new_width:.8f} "
                f"{new_height:.8f}"
            )

            corrected_lines.append(corrected_line)

        # Save the corrected label file
        label_path.write_text(
            "\n".join(corrected_lines) + "\n",
            encoding="utf-8",
        )

        fixed_count += 1

        print(f"Corrected: {split}/{label_path.name}")

print(f"\nCorrection completed. Files processed: {fixed_count}")
