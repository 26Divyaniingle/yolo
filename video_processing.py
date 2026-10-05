import cv2
import time
cap = cv2.VideoCapture("videos/input.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "output/output.mp4",
    fourcc,
    video_fps,
    (width, height),
    False
)

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_time = time.time()

    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        gray,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        255,
        2
    )

    cv2.imshow("Grayscale Video", gray)
    out.write(gray)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Step 14: Release resources
cap.release()
out.release()

# Step 15: Close windows
cv2.destroyAllWindows()

print("Processing complete!")
print("Saved as: output/output.mp4")