import cv2

image = cv2.imread("images/test.jpg")

print(image)
print(image.shape)
if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")
    
#cv2.imshow("Original Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output/copied_image.jpg", image)
print("Image saved successfully")

#resize

resized = cv2.resize(image, (640, 640))

print("Original:", image.shape)
print("Resized:", resized.shape)

#cv2.imshow("Original", image)
#cv2.imshow("Resized", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()


crop = image[100:500, 100:500]
#cv2.imshow("Crop", crop)

cv2.waitKey(0)
cv2.destroyAllWindows()

rotated = cv2.rotate(
    image,
    cv2.ROTATE_180
)

#cv2.imshow("Rotated", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()

#flipped = cv2.flip(image, 1)
flipped = cv2.flip(image, 0)
#cv2.imshow("Flipped", flipped)

cv2.waitKey(0)
cv2.destroyAllWindows()

#blurred = cv2.GaussianBlur(image, (8, 8), 0)
median = cv2.medianBlur(image, 5)
#cv2.imshow("Original", image)
#cv2.imshow("median", median)

cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

#cv2.imshow("Gray", gray)
#cv2.imshow("Threshold", threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 200)

#cv2.imshow("Original", image)
#cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

output = image.copy()

cv2.rectangle(
    output,
    (50, 50),
    (500, 400),
    (255, 0, 0),
    2
)

#cv2.imshow("Bounding Box", output)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.putText(
    output,
    "FLOWER",
    (100, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Detection", output)

cv2.waitKey(0)
cv2.destroyAllWindows()