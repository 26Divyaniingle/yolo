from pathlib import Path


# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Original downloaded dataset
RAW_DIR = PROJECT_DIR / "raw_dataset"


def inspect_dataset():
    print("\n========== DATASET INSPECTION ==========\n")

    if not RAW_DIR.exists():
        print("ERROR: raw_dataset folder does not exist.")
        return

    # Show files and folders near the dataset root
    print("Raw dataset path:")
    print(RAW_DIR)

    print("\nTop-level contents:")

    for item in RAW_DIR.iterdir():
        if item.is_dir():
            print(f"[FOLDER] {item.name}")
        else:
            print(f"[FILE]   {item.name}")

    # Count image files
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    images = [
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in image_extensions
    ]

    # Count annotation formats
    xml_files = list(RAW_DIR.rglob("*.xml"))
    json_files = list(RAW_DIR.rglob("*.json"))
    txt_files = list(RAW_DIR.rglob("*.txt"))

    print("\n========== FILE COUNTS ==========")
    print("Images:", len(images))
    print("XML annotation files:", len(xml_files))
    print("JSON files:", len(json_files))
    print("TXT files:", len(txt_files))

    print("\n========== SAMPLE IMAGES ==========")

    for image in images[:10]:
        print(image.relative_to(RAW_DIR))

    print("\n========== SAMPLE XML FILES ==========")

    for file in xml_files[:10]:
        print(file.relative_to(RAW_DIR))

    print("\n========== SAMPLE JSON FILES ==========")

    for file in json_files[:10]:
        print(file.relative_to(RAW_DIR))

    print("\n========== SAMPLE TXT FILES ==========")

    for file in txt_files[:10]:
        print(file.relative_to(RAW_DIR))

    print("\nInspection completed.")


if __name__ == "__main__":
    inspect_dataset()