from pathlib import Path
import xml.etree.ElementTree as ET


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "raw_dataset"

OUTPUT_DIR = PROJECT_DIR / "dataset"

OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_LABELS_DIR = OUTPUT_DIR / "labels"


# --------------------------------------------------
# 2. Class names
# --------------------------------------------------

# IMPORTANT:
# Replace these example class names with the exact
# class names found in your dataset's XML files.
#
# The order determines the YOLO class IDs.

CLASS_NAMES = [
    "With Helmet",
    "Without Helmet",
]


CLASS_TO_ID = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# --------------------------------------------------
# 3. Image extensions
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# --------------------------------------------------
# 4. Find images by filename
# --------------------------------------------------

def find_image(filename):

    matches = [
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and path.name == filename
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not matches:
        return None

    if len(matches) > 1:
        print(
            f"WARNING: Multiple images named {filename}. "
            "Using the first match."
        )

    return matches[0]


# --------------------------------------------------
# 5. Convert pixel boxes into YOLO coordinates
# --------------------------------------------------

def convert_box(xmin, ymin, xmax, ymax, image_width, image_height):

    box_width = xmax - xmin
    box_height = ymax - ymin

    x_center = xmin + box_width / 2
    y_center = ymin + box_height / 2

    # Normalize values between 0 and 1
    x_center /= image_width
    y_center /= image_height

    box_width /= image_width
    box_height /= image_height

    return x_center, y_center, box_width, box_height


# --------------------------------------------------
# 6. Convert one XML file
# --------------------------------------------------

def convert_xml(xml_path):

    root = ET.parse(xml_path).getroot()

    filename = root.findtext("filename")

    if not filename:
        print(f"SKIPPED: filename missing in {xml_path.name}")
        return False

    image_path = find_image(filename)

    if image_path is None:
        print(f"SKIPPED: image not found: {filename}")
        return False

    size = root.find("size")

    if size is None:
        print(f"SKIPPED: image size missing: {xml_path.name}")
        return False

    image_width = int(size.findtext("width"))
    image_height = int(size.findtext("height"))

    if image_width <= 0 or image_height <= 0:
        print(f"SKIPPED: invalid dimensions: {xml_path.name}")
        return False

    yolo_lines = []

    for obj in root.findall("object"):

        class_name = obj.findtext("name")

        if class_name not in CLASS_TO_ID:
            print(
                f"WARNING: Unknown class '{class_name}' "
                f"in {xml_path.name}; skipping this object."
            )
            continue

        box = obj.find("bndbox")

        if box is None:
            print(f"WARNING: Missing bndbox in {xml_path.name}")
            continue

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        # Check the original coordinates
        if not (
            0 <= xmin < xmax <= image_width
            and 0 <= ymin < ymax <= image_height
        ):
            print(
                f"WARNING: Invalid box in {xml_path.name}: "
                f"{xmin}, {ymin}, {xmax}, {ymax}"
            )
            continue

        x_center, y_center, width, height = convert_box(
            xmin,
            ymin,
            xmax,
            ymax,
            image_width,
            image_height
        )

        class_id = CLASS_TO_ID[class_name]

        yolo_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    # Copy the image into the new dataset
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    import shutil

    destination_image = OUTPUT_IMAGES_DIR / image_path.name

    shutil.copy2(image_path, destination_image)

    # Save YOLO label file
    label_path = OUTPUT_LABELS_DIR / f"{image_path.stem}.txt"

    label_path.write_text(
        "\n".join(yolo_lines),
        encoding="utf-8"
    )

    return True


# --------------------------------------------------
# 7. Convert the dataset
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = list(RAW_DIR.rglob("*.xml"))

    if not xml_files:
        print("ERROR: No XML annotation files found.")
        print("Use this converter only for Pascal VOC XML.")
        return

    successful = 0

    for xml_path in xml_files:

        result = convert_xml(xml_path)

        if result:
            successful += 1

    print("\n========== CONVERSION SUMMARY ==========")
    print("XML files found:", len(xml_files))
    print("Successfully processed:", successful)
    print("Output images:", OUTPUT_IMAGES_DIR)
    print("Output labels:", OUTPUT_LABELS_DIR)


if __name__ == "__main__":
    main()