from ultralytics import YOLO
import cv2
from pathlib import Path

# Base directory of this script
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR if (BASE_DIR / "videos").exists() else BASE_DIR.parent

# Pretrained segmentation model
MODEL_PATH = str(BASE_DIR / "yolo11n-seg.pt") if (BASE_DIR / "yolo11n-seg.pt").exists() else "yolo11n-seg.pt"
model = YOLO(MODEL_PATH)

# Dynamically find traffic.mp4 (checks week1day4/video/ and project root videos/)
if (BASE_DIR / "video" / "traffic.mp4").exists():
    video_path = str(BASE_DIR / "video" / "traffic.mp4")
elif (PROJECT_ROOT / "videos" / "traffic.mp4").exists():
    video_path = str(PROJECT_ROOT / "videos" / "traffic.mp4")
else:
    video_path = "video/traffic.mp4"

cap = cv2.VideoCapture(video_path)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError(f"Video ka frame read nahi hua from: {video_path}")

results = model(frame, verbose=False)

annotated_frame = results[0].plot()

# Scale down so 4K vertical frame fits comfortably on your screen
h, w = annotated_frame.shape[:2]
disp_h = 800
disp_w = int(w * (disp_h / h))
display_frame = cv2.resize(annotated_frame, (disp_w, disp_h))

cv2.imshow("YOLO Segmentation", display_frame)

cv2.waitKey(0)

cv2.destroyAllWindows()