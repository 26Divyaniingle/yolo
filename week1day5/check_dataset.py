from pathlib import Path
import yaml


# --------------------------------------------------
# 1. Locate the YAML
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

YAML_PATH = PROJECT_DIR / "yolo_dataset" / "data.yaml"


if not YAML_PATH.exists():
    raise FileNotFoundError(
        f"data.yaml not found: {YAML_PATH}"
    )


# --------------------------------------------------
# 2. Read the YAML
# --------------------------------------------------

with open(YAML_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


print("\n========== DATASET CONFIG ==========")

print("YAML:", YAML_PATH)

print("\nClasses:")

for class_id, class_name in config["names"].items():
    print(f"{class_id}: {class_name}")


# --------------------------------------------------
# 3. Resolve dataset root
# --------------------------------------------------

dataset_root = Path(config["path"])

if not dataset_root.is_absolute():
    dataset_root = YAML_PATH.parent / dataset_root

dataset_root = dataset_root.resolve()

print("\nDataset root:", dataset_root)


# --------------------------------------------------
# 4. Check image/label pairs
# --------------------------------------------------

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


for split in ["train", "val", "test"]:

    relative_path = config.get(split)

    if not relative_path:
        print(f"\n{split}: Not configured")
        continue

    image_dir = dataset_root / relative_path

    # Replace images with labels
    label_dir = Path(
        str(image_dir).replace(
            f"{Path('images') / split}",
            f"{Path('labels') / split}"
        )
    )

    print(f"\n========== {split.upper()} ==========")

    print("Image directory:", image_dir)
    print("Label directory:", label_dir)

    if not image_dir.exists():
        print("ERROR: Image directory does not exist.")
        continue

    if not label_dir.exists():
        print("ERROR: Label directory does not exist.")
        continue

    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in image_extensions
    ]

    missing_labels = []

    for image in images:

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            missing_labels.append(image.name)

    print("Images:", len(images))
    print("Images without labels:", len(missing_labels))

    if missing_labels:
        print("First missing labels:")

        for filename in missing_labels[:10]:
            print(" -", filename)


print("\nDataset check completed.")