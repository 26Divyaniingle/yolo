from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "yolo_dataset"

LABEL_DIR = DATASET_DIR / "labels"

NUM_CLASSES = 2


def validate_label_file(label_path):

    errors = []

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):

        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            errors.append(
                f"Line {line_number}: expected 5 values"
            )
            continue

        try:
            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

        except ValueError:
            errors.append(
                f"Line {line_number}: non-numeric value"
            )
            continue

        if not 0 <= class_id < NUM_CLASSES:
            errors.append(
                f"Line {line_number}: invalid class ID"
            )

        if not all(
            0 <= value <= 1
            for value in [
                x_center,
                y_center,
                width,
                height
            ]
        ):
            errors.append(
                f"Line {line_number}: coordinate outside 0-1"
            )

        if width <= 0 or height <= 0:
            errors.append(
                f"Line {line_number}: box width/height must be positive"
            )

        if (
            x_center - width / 2 < 0
            or x_center + width / 2 > 1
            or y_center - height / 2 < 0
            or y_center + height / 2 > 1
        ):
            errors.append(
                f"Line {line_number}: bounding box extends outside image"
            )

    return errors


def main():

    total_files = 0
    total_errors = 0

    for split in ["train", "val", "test"]:

        split_dir = LABEL_DIR / split

        if not split_dir.exists():
            print("Missing directory:", split_dir)
            continue

        label_files = list(split_dir.glob("*.txt"))

        print(f"\n{split.upper()}")

        for label_path in label_files:

            errors = validate_label_file(label_path)

            total_files += 1

            if errors:

                total_errors += len(errors)

                print("\n", label_path.name)

                for error in errors:
                    print("  ERROR:", error)

        print("Label files:", len(label_files))

    print("\n========== SUMMARY ==========")
    print("Files checked:", total_files)
    print("Errors found:", total_errors)


if __name__ == "__main__":
    main()