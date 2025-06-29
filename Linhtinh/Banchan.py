import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "feet-outline-paper-crafts.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding to isolate the black contour
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours from the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty image to draw the contours
contour_image = np.zeros_like(image)
cv2.drawContours(contour_image, contours, -1, (255), thickness=2)

# Show the result

thicker_contour_image = np.zeros_like(image)
cv2.drawContours(thicker_contour_image, contours, -1, (255), thickness=5)

# Hiển thị kết quả với viền dày hơn
plt.figure(figsize=(6, 10))
plt.title("Thicker Contour")
plt.imshow(thicker_contour_image, cmap='gray')
plt.axis('off')
plt.show()