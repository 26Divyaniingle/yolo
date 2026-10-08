import cv2
import time
import os
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 1. CONFIGURATION & PATH RESOLUTION
# ============================================================

# Automatically locate project root whether run from root or inside week1day4
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR if (BASE_DIR / "videos").exists() else BASE_DIR.parent

MODEL_PATH = str(PROJECT_ROOT / "yolo11n.pt")
VIDEO_PATH = str(PROJECT_ROOT / "videos" / "traffic.mp4")
OUTPUT_PATH = str(PROJECT_ROOT / "output" / "tracked_output.mp4")

CONFIDENCE = 0.30

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ============================================================
# 2. LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully.")

# ============================================================
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Video open nahi hui: {VIDEO_PATH}")

# ============================================================
# 4. READ VIDEO PROPERTIES & SCALING SETUP
# ============================================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

if video_fps <= 0:
    video_fps = 30.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video width  : {width}")
print(f"Video height : {height}")
print(f"Video FPS    : {video_fps:.2f}")
print(f"Total frames : {total_frames}")

# COUNTING LINE:
# Note: traffic.mp4 is 2160x3840. The upper portion (0 to ~1900) is sky.
# The road is located between y=1900 and y=3840.
# We place the counting line at ~65% of height (y = 2496) across the road.
LINE_Y = int(height * 0.65)

# DISPLAY RESIZE:
# Scale down the 4K display frame so it fits standard desktop monitors
DISPLAY_HEIGHT = 800
display_scale = DISPLAY_HEIGHT / height
display_width = int(width * display_scale)

# UI scale factor for text and line thickness on the 4K canvas
scale_factor = max(1.0, height / 1080.0)

# ============================================================
# 5. CREATE OUTPUT VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    video_fps,
    (width, height)
)

# ============================================================
# 6. TRACKING CLASSES & DATA
# ============================================================

# Target COCO classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
PERSON_CLASS = 0
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TARGET_CLASSES = [PERSON_CLASS] + sorted(list(VEHICLE_CLASSES.keys()))

# Unique IDs that crossed the line
crossed_person_ids = set()
crossed_vehicle_ids = set()

# Previous Y position of each track ID
previous_y = {}

# FPS calculation variables
prev_time = time.time()
fps = 0.0
frame_number = 0

print("=" * 60)
print("LINE COUNTER & TRACKER STARTED")
print(f"Counting line placed on road at Y = {LINE_Y}")
print(f"Display window scaled to {display_width}x{DISPLAY_HEIGHT}")
print("=" * 60)

# ============================================================
# 7. PROCESS VIDEO FRAME BY FRAME
# ============================================================

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_number += 1

    # --------------------------------------------------------
    # Calculate real-time FPS
    # --------------------------------------------------------
    curr_time = time.time()
    time_diff = curr_time - prev_time
    if time_diff > 0:
        fps = 1.0 / time_diff
    prev_time = curr_time

    # --------------------------------------------------------
    # Track persons and vehicles using YOLO + ByteTrack
    # --------------------------------------------------------
    results = model.track(
        frame,
        persist=True,
        classes=TARGET_CLASSES,
        conf=CONFIDENCE,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # Draw Counting Line on the road
    # --------------------------------------------------------
    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (0, 0, 255),
        int(4 * scale_factor)
    )
    cv2.putText(
        frame,
        "COUNTING LINE",
        (int(40 * scale_factor), LINE_Y - int(15 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8 * scale_factor,
        (0, 0, 255),
        int(2 * scale_factor)
    )

    # --------------------------------------------------------
    # Read detections, draw bounding boxes & check line crossing
    # --------------------------------------------------------
    current_persons = 0
    current_vehicles = 0

    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(xyxy, track_ids, class_ids, confs):
            x1, y1, x2, y2 = map(int, box)

            # In-frame count
            if cls_id == PERSON_CLASS:
                current_persons += 1
                cls_name = "Person"
                box_color = (0, 255, 0)      # Green for persons
            elif cls_id in VEHICLE_CLASSES:
                current_vehicles += 1
                cls_name = VEHICLE_CLASSES[cls_id].capitalize()
                box_color = (255, 200, 0)    # Cyan/Orange for vehicles
            else:
                continue

            # Bottom center point (contact with road: feet or tires)
            center_x = (x1 + x2) // 2
            center_y = y2

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                int(2 * scale_factor)
            )

            # Draw contact point
            cv2.circle(
                frame,
                (center_x, center_y),
                int(5 * scale_factor),
                (0, 0, 255),
                -1
            )

            # Draw tracking ID and label
            label = f"{cls_name} ID:{track_id} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - int(10 * scale_factor), int(25 * scale_factor))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65 * scale_factor,
                box_color,
                int(2 * scale_factor)
            )

            # Check line crossing (top-to-bottom or bottom-to-top)
            if track_id in previous_y:
                old_y = previous_y[track_id]
                crossed = (old_y < LINE_Y <= center_y) or (center_y <= LINE_Y < old_y)

                if crossed:
                    if cls_id == PERSON_CLASS:
                        crossed_person_ids.add(track_id)
                    elif cls_id in VEHICLE_CLASSES:
                        crossed_vehicle_ids.add(track_id)

            previous_y[track_id] = center_y

    # --------------------------------------------------------
    # Draw HUD Overlay Card (FPS, Crossings, In-Frame Counts)
    # --------------------------------------------------------
    overlay = frame.copy()
    hud_w = int(580 * scale_factor)
    hud_h = int(270 * scale_factor)
    cv2.rectangle(
        overlay,
        (20, 20),
        (20 + hud_w, 20 + hud_h),
        (20, 20, 20),
        -1
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        "TRAFFIC & PERSON TRACKER",
        (int(40 * scale_factor), int(60 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9 * scale_factor,
        (0, 255, 255),
        int(2 * scale_factor)
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (int(40 * scale_factor), int(105 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75 * scale_factor,
        (255, 255, 255),
        int(2 * scale_factor)
    )
    cv2.putText(
        frame,
        f"Unique Persons Crossed: {len(crossed_person_ids)}",
        (int(40 * scale_factor), int(145 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75 * scale_factor,
        (0, 255, 0),
        int(2 * scale_factor)
    )
    cv2.putText(
        frame,
        f"Vehicles Crossed: {len(crossed_vehicle_ids)}",
        (int(40 * scale_factor), int(185 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75 * scale_factor,
        (255, 200, 0),
        int(2 * scale_factor)
    )
    cv2.putText(
        frame,
        f"In Frame -> Persons: {current_persons} | Vehicles: {current_vehicles}",
        (int(40 * scale_factor), int(225 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65 * scale_factor,
        (200, 200, 200),
        int(2 * scale_factor)
    )
    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (int(40 * scale_factor), int(260 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * scale_factor,
        (180, 180, 180),
        int(2 * scale_factor)
    )

    # --------------------------------------------------------
    # Save Full-Resolution Frame to Output Video
    # --------------------------------------------------------
    writer.write(frame)

    # --------------------------------------------------------
    # Display Resized Frame (fits monitor screen)
    # --------------------------------------------------------
    display_frame = cv2.resize(frame, (display_width, DISPLAY_HEIGHT))
    cv2.imshow("YOLO Unique Counter & Tracker", display_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped by user.")
        break

# ============================================================
# 8. CLEANUP & RELEASE RESOURCES
# ============================================================

cap.release()
writer.release()
cv2.destroyAllWindows()

# ============================================================
# 9. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total Frames Processed : {frame_number}")
print(f"Unique Persons Crossed : {len(crossed_person_ids)}")
print(f"Vehicles Crossed       : {len(crossed_vehicle_ids)}")
print(f"Output saved at        : {OUTPUT_PATH}")
print("=" * 60)