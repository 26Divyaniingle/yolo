import cv2
import time
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "yolo26n.pt"

VIDEO_PATH = "videos/traffic.mp4"

OUTPUT_PATH = "output\d4s3.mp4"

CONFIDENCE_THRESHOLD = 0.30

# YOLO inference image size
IMAGE_SIZE = 640


# ============================================================
# 2. DEFINE CLASSES
# ============================================================

# COCO class IDs used by the pretrained YOLO detection model

PERSON_CLASS = 0

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# ============================================================
# 3. LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# 4. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)


if not cap.isOpened():

    print("ERROR: Could not open video.")

    exit()


# ============================================================
# 5. GET VIDEO INFORMATION
# ============================================================

video_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

video_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print()
print("=" * 60)
print("VIDEO INFORMATION")
print("=" * 60)

print(f"Width        : {video_width}")
print(f"Height       : {video_height}")
print(f"Video FPS    : {video_fps:.2f}")
print(f"Total Frames : {total_frames}")

print("=" * 60)


# ============================================================
# 6. CREATE VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    video_fps,
    (video_width, video_height)
)


# ============================================================
# 7. FPS VARIABLES
# ============================================================

previous_time = time.time()

fps = 0.0


# ============================================================
# 8. FRAME COUNTER
# ============================================================

frame_number = 0


# ============================================================
# 9. MAIN VIDEO LOOP
# ============================================================

while cap.isOpened():

    # --------------------------------------------------------
    # Read one frame
    # --------------------------------------------------------

    success, frame = cap.read()


    # --------------------------------------------------------
    # Stop when video ends
    # --------------------------------------------------------

    if not success:

        print("Video finished.")

        break


    frame_number += 1


    # ========================================================
    # 10. YOLO INFERENCE
    # ========================================================

    results = model(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    # ========================================================
    # 11. GET RESULT
    # ========================================================

    result = results[0]

    boxes = result.boxes

    names = result.names


    # ========================================================
    # 12. INITIALIZE COUNTERS
    # ========================================================

    person_count = 0

    vehicle_count = 0

    car_count = 0

    motorcycle_count = 0

    bus_count = 0

    truck_count = 0

    bicycle_count = 0


    # ========================================================
    # 13. PROCESS DETECTIONS
    # ========================================================

    if boxes is not None and len(boxes) > 0:

        for i in range(len(boxes)):

            # ------------------------------------------------
            # Get class ID
            # ------------------------------------------------

            class_id = int(
                boxes.cls[i].item()
            )


            # ------------------------------------------------
            # Get confidence
            # ------------------------------------------------

            confidence = float(
                boxes.conf[i].item()
            )


            # ------------------------------------------------
            # Get bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                boxes.xyxy[i].tolist()
            )


            # ------------------------------------------------
            # Convert coordinates to integers
            # ------------------------------------------------

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)


            # ------------------------------------------------
            # Get class name
            # ------------------------------------------------

            class_name = names[class_id]


            # =================================================
            # 14. PERSON
            # =================================================

            if class_id == PERSON_CLASS:

                person_count += 1


            # =================================================
            # 15. VEHICLES
            # =================================================

            elif class_id in VEHICLE_CLASSES:

                vehicle_count += 1


                if class_id == 1:

                    bicycle_count += 1

                elif class_id == 2:

                    car_count += 1

                elif class_id == 3:

                    motorcycle_count += 1

                elif class_id == 5:

                    bus_count += 1

                elif class_id == 7:

                    truck_count += 1


            # =================================================
            # 16. DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # 17. CUSTOM LABEL
            # =================================================

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # ========================================================
    # 18. CALCULATE FPS
    # ========================================================

    current_time = time.time()

    time_difference = (
        current_time - previous_time
    )


    if time_difference > 0:

        fps = 1 / time_difference


    previous_time = current_time


    # ========================================================
    # 19. DRAW FPS
    # ========================================================

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


    # ========================================================
    # 20. DRAW PERSON COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"Persons: {person_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 21. DRAW VEHICLE COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 22. DRAW VEHICLE BREAKDOWN
    # ========================================================

    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Motorcycles: {motorcycle_count}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Buses: {bus_count}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Trucks: {truck_count}",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 23. DRAW FRAME NUMBER
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, video_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 24. DISPLAY FRAME
    # ========================================================

    cv2.imshow(
        "YOLO Real-Time Person & Vehicle Counter",
        frame
    )


    # ========================================================
    # 25. SAVE FRAME TO OUTPUT VIDEO
    # ========================================================

    writer.write(frame)


    # ========================================================
    # 26. PRESS Q TO EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print("Stopped by user.")

        break


# ============================================================
# 27. RELEASE RESOURCES
# ============================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


# ============================================================
# 28. FINAL MESSAGE
# ============================================================

print()
print("=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

print(f"Frames processed : {frame_number}")

print(f"Output video     : {OUTPUT_PATH}")

print("Done!")