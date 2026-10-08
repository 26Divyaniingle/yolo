from pathlib import Path
import cv2


PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_DIR / "yolo_dataset" / "images" / "train"

LABEL_DIR = PROJECT_DIR / "yolo_dataset" / "labels" / "train"

CLASS_NAMES = [
    "With Helmet",
    "No Helmet",
]


def preview_image(image_path):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        return

    image_height, image_width = image.shape[:2]

    if not label_path.exists():
        print("Label missing:", label_path)
        return

    with open(label_path, "r", encoding="utf-8") as file:

        for line in file:

            values = line.split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1]) * image_width
            y_center = float(values[2]) * image_height

            box_width = float(values[3]) * image_width
            box_height = float(values[4]) * image_height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)

            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Keep the box inside the image
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))

            if 0 <= class_id < len(CLASS_NAMES):

                label = CLASS_NAMES[class_id]

            else:

                label = f"class_{class_id}"

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO Annotation Preview", image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def main():

    images = [
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    ]

    if not images:
        print("No images found:", IMAGE_DIR)
        return

    for image_path in images[:20]:

        print("Previewing:", image_path.name)

        preview_image(image_path)


if __name__ == "__main__":
    main()