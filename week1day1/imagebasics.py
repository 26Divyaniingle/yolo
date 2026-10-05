import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/test.jpg")


print("Image loaded successfully")
print("Shape:", image.shape)

height = image.shape[0]
width = image.shape[1]
channels = image.shape[2]

print("Height:", height)
print("Width:", width)
print("Channels:", channels)

total_pixels = height * width

print("Total pixels:", total_pixels)

print("Data type:", image.dtype)

pixel = image[100, 200]

print("Pixel at row 100, column 200:", pixel)

blue = image[:, :, 0]
green = image[:, :, 1]
red = image[:, :, 2]

print("Blue channel shape:", blue.shape)
print("Green channel shape:", green.shape)
print("Red channel shape:", red.shape)

#cv2.imshow("Original Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

#BGR --> RBG

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("BGR pixel:", image[100, 200])
print("RGB pixel:", rgb_image[100, 200])


#BGR TO GRAYSCALE

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Color shape:", image.shape)
print("Grayscale shape:", gray_image.shape)


#BGR TO HSV

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print("HSV shape:", hsv_image.shape)

#all 4images


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(rgb_image)
plt.title("RGB")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(hsv_image)
plt.title("HSV array")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(image)
plt.title("OpenCV BGR shown directly")
plt.axis("off")

plt.show()