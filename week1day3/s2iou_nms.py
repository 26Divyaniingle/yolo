import numpy as np
import cv2


#iou
def calculate_iou(box1, box2):

    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection width and height
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    # Intersection area
    intersection_area = (
        intersection_width * intersection_height
    )

    # Box 1 area
    area_box1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    # Box 2 area
    area_box2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    # Union area
    union_area = (
        area_box1 +
        area_box2 -
        intersection_area
    )

    if union_area == 0:
        return 0

    return intersection_area / union_area


# ============================================================
# 2. NMS FROM SCRATCH
# ============================================================

def nms_from_scratch(boxes, scores, iou_threshold):

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Sort scores from highest to lowest
    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:

        # Highest confidence box
        current = order[0]

        # Keep it
        keep.append(current)

        remaining = []

        # Compare with remaining boxes
        for index in order[1:]:

            iou = calculate_iou(
                boxes[current],
                boxes[index]
            )

            # Keep if overlap is not too high
            if iou <= iou_threshold:
                remaining.append(index)

        order = np.array(remaining)

    return keep


# ============================================================
# 3. TEST IOU
# ============================================================

box1 = [100, 100, 300, 300]
box2 = [150, 150, 350, 350]

iou = calculate_iou(box1, box2)

print("=" * 50)
print("IOU TEST")
print("=" * 50)

print("Box 1:", box1)
print("Box 2:", box2)
print("IoU:", iou)


# ============================================================
# 4. TEST NMS
# ============================================================

boxes = [
    [100, 100, 300, 300],
    [120, 120, 310, 310],
    [500, 500, 700, 700]
]

scores = [
    0.92,
    0.87,
    0.80
]

iou_threshold = 0.5


# ============================================================
# 5. OUR NMS
# ============================================================

our_result = nms_from_scratch(
    boxes,
    scores,
    iou_threshold
)

print("\n" + "=" * 50)
print("OUR NMS")
print("=" * 50)

print("Kept indices:", our_result)

for index in our_result:
    print(
        "Box:",
        boxes[index],
        "| Score:",
        scores[index]
    )


# ============================================================
# 6. OPENCV NMS
# ============================================================

opencv_result = cv2.dnn.NMSBoxes(
    boxes,
    scores,
    score_threshold=0.0,
    nms_threshold=iou_threshold
)

opencv_result = np.array(
    opencv_result
).flatten().tolist()


print("\n" + "=" * 50)
print("OPENCV NMS")
print("=" * 50)

print("Kept indices:", opencv_result)

for index in opencv_result:
    print(
        "Box:",
        boxes[index],
        "| Score:",
        scores[index]
    )


# ============================================================
# 7. COMPARISON
# ============================================================

print("\n" + "=" * 50)
print("COMPARISON")
print("=" * 50)

print("Our NMS:", our_result)
print("OpenCV NMS:", opencv_result)