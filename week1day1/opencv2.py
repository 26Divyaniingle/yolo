import cv2
import os

input_folder = "images"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    input_path = os.path.join(input_folder, filename)

    image = cv2.imread(input_path)

    if image is None:
        continue

    # Resize
    image = cv2.resize(image, (640, 640))

    # Draw rectangle
    cv2.rectangle(
        image,
        (100, 100),
        (500, 500),
        (0, 255, 0),
        2
    )

    # Add label
    cv2.putText(
        image,
        "Object",
        (100, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Save
    output_path = os.path.join(
        output_folder,
        filename
    )

    cv2.imwrite(output_path, image)

    print("Processed:", filename)

print("All images processed!")