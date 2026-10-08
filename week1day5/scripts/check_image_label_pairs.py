from pathlib import Path
import yaml

# --------------------------------------------------
# 1. Find project and YAML paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
YAML_PATH = PROJECT_DIR / "yolo_dataset" / "data.yaml"

if not YAML_PATH.exists():
    raise FileNotFoundError(
        f"data.yaml was not found: {YAML_PATH}"
    )

with open(YAML_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

dataset_root = Path(config["path"])

if not dataset_root.is_absolute():
    dataset_root = (YAML_PATH.parent / dataset_root).resolve()

print("=" * 60)
print("YOLO DATASET IMAGE-LABEL VALIDATION")
print("=" * 60)

print("Dataset root:", dataset_root)


# --------------------------------------------------
# 2. Supported image extensions
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# --------------------------------------------------
# 3. Check each dataset split
# --------------------------------------------------

total_images = 0
total_labels = 0
total_missing_labels = 0
total_orphan_labels = 0
total_empty_labels = 0

split_counts = {}

for split in ["train", "val", "test"]:

    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    print("\n" + "-" * 60)
    print(f"SPLIT: {split.upper()}")
    print("-" * 60)

    if not image_dir.exists():
        print("ERROR: Image directory not found:", image_dir)
        continue

    if not label_dir.exists():
        print("ERROR: Label directory not found:", label_dir)
        continue

    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    # Match files using the filename without extension.
    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    empty_labels = [
        label
        for label in labels
        if not label.read_text(
            encoding="utf-8"
        ).strip()
    ]

    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Images without label files:", len(missing_labels))
    print("Labels without matching images:", len(orphan_labels))
    print("Empty label files:", len(empty_labels))

    if missing_labels:
        print("\nImages without labels (first 10):")
        for name in sorted(missing_labels)[:10]:
            print("  ", name)

    if orphan_labels:
        print("\nLabels without images (first 10):")
        for name in sorted(orphan_labels)[:10]:
            print("  ", name)

    if empty_labels:
        print("\nEmpty labels (first 10):")
        for label in empty_labels[:10]:
            print("  ", label.name)

    split_counts[split] = len(images)

    total_images += len(images)
    total_labels += len(labels)
    total_missing_labels += len(missing_labels)
    total_orphan_labels += len(orphan_labels)
    total_empty_labels += len(empty_labels)


# --------------------------------------------------
# 4. Report image split percentages
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("Total images:", total_images)
print("Total label files:", total_labels)
print("Missing label files:", total_missing_labels)
print("Orphan label files:", total_orphan_labels)
print("Empty label files:", total_empty_labels)

print("\nImage split percentages:")

for split, count in split_counts.items():

    percentage = (
        count / total_images * 100
        if total_images > 0
        else 0
    )

    print(f"{split.upper():5}: {count:4} images ({percentage:.2f}%)")


# --------------------------------------------------
# 5. Final status
# --------------------------------------------------

if total_images == 0:
    print("\nSTATUS: FAILED - No images found.")

elif total_missing_labels > 0 or total_orphan_labels > 0:
    print(
        "\nSTATUS: REVIEW REQUIRED - "
        "Some image-label pairs do not match."
    )

else:
    print(
        "\nSTATUS: IMAGE-LABEL PAIRING CHECK PASSED."
    )

print("\nNote: Empty label files are reported separately.")
print("Verify whether these images genuinely contain no objects.")
print("This script checks pairing, not label coordinate validity.")

