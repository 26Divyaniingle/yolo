def xyxy_to_yolo(x1, y1, x2, y2, image_width, image_height):
    """
    Convert bounding box from xyxy format
    to normalized YOLO format.
    """

    # Step 1: Calculate center coordinates
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Step 2: Calculate bounding box width and height
    box_width = x2 - x1
    box_height = y2 - y1

    # Step 3: Normalize the values
    x_center = x_center / image_width
    y_center = y_center / image_height
    box_width = box_width / image_width
    box_height = box_height / image_height

    return x_center, y_center, box_width, box_height


# --------------------------------
# IMAGE INFORMATION
# --------------------------------

image_width = 800
image_height = 600


# --------------------------------
# BOUNDING BOXES
# --------------------------------
# Format:
# (class_id, x1, y1, x2, y2)

boxes = [
    (0, 100, 150, 500, 450),
    (1, 50, 100, 300, 350),
    (2, 400, 200, 700, 500)
]


# --------------------------------
# CONVERT EACH BOX
# --------------------------------

yolo_lines = []

for class_id, x1, y1, x2, y2 in boxes:

    x_center, y_center, width, height = xyxy_to_yolo(
        x1,
        y1,
        x2,
        y2,
        image_width,
        image_height
    )

    # Create one YOLO annotation line
    line = (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )

    yolo_lines.append(line)


# --------------------------------
# SAVE TO TXT FILE
# --------------------------------

output_file = "output/image1.txt"

with open(output_file, "w") as file:
    for line in yolo_lines:
        file.write(line + "\n")


print("Conversion completed!")
print(f"YOLO annotation saved to: {output_file}")

print("\nYOLO annotations:")

for line in yolo_lines:
    print(line)