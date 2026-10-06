def xyxy_to_yolo(x1, y1, x2, y2, image_width, image_height):
    # Calculate center
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Calculate width and height
    box_width = x2 - x1
    box_height = y2 - y1

    # Normalize
    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height

    return x_center, y_center, box_width, box_height


# Image dimensions
image_width = 800
image_height = 600

# Class ID
class_id = 0

# Bounding box in xyxy format
x1 = 100
y1 = 150
x2 = 500
y2 = 450

# Convert
x_center, y_center, width, height = xyxy_to_yolo(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height
)

# Print YOLO format
print(
    class_id,
    f"{x_center:.6f}",
    f"{y_center:.6f}",
    f"{width:.6f}",
    f"{height:.6f}"
)