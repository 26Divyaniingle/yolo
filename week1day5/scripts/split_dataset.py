from pathlib import Path
import random
import shutil


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_IMAGES = PROJECT_DIR / "dataset" / "images"
SOURCE_LABELS = PROJECT_DIR / "dataset" / "labels"

OUTPUT_DIR = PROJECT_DIR / "yolo_dataset"


# --------------------------------------------------
# 2. Settings
# --------------------------------------------------

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# --------------------------------------------------
# 3. Collect images
# --------------------------------------------------

images = [
    path
    for path in SOURCE_IMAGES.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
]


if not images:
    raise RuntimeError(
        f"No images found in {SOURCE_IMAGES}"
    )


# Keep only images that have corresponding labels
paired_images = []

for image_path in images:

    label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

    if label_path.exists():
        paired_images.append(image_path)
    else:
        print("Skipping image without label:", image_path.name)


if not paired_images:
    raise RuntimeError("No image/label pairs found.")


# --------------------------------------------------
# 4. Shuffle and split
# --------------------------------------------------

random.seed(SEED)

random.shuffle(paired_images)

total = len(paired_images)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": paired_images[:train_end],
    "val": paired_images[train_end:val_end],
    "test": paired_images[val_end:]
}


# --------------------------------------------------
# 5. Copy images and labels
# --------------------------------------------------

for split_name, split_images in splits.items():

    image_output = (
        OUTPUT_DIR / "images" / split_name
    )

    label_output = (
        OUTPUT_DIR / "labels" / split_name
    )

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    for image_path in split_images:

        label_path = (
            SOURCE_LABELS / f"{image_path.stem}.txt"
        )

        shutil.copy2(
            image_path,
            image_output / image_path.name
        )

        shutil.copy2(
            label_path,
            label_output / label_path.name
        )

    print(
        f"{split_name}: {len(split_images)} images"
    )


print("\nDataset splitting completed.")
print("Output:", OUTPUT_DIR)