import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read Image
filename = "sample_image.jpg"   # You can use any image
image = cv2.imread(filename)

if image is None:
    print("Error: Image not found. Please check the file name and path.")
    exit()

# Convert from BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Display basic info
print("=== Image Information ===")
print(f"Shape (Height, Width, Channels): {image_rgb.shape}")
print(f"Data Type: {image_rgb.dtype}")
print(f"Total Pixels: {image_rgb.size}")
print("==========================\n")

# Step 3: Extract basic color features
mean_color = cv2.mean(image_rgb)[:3]  # Mean intensity per channel
std_color = image_rgb.std(axis=(0, 1))  # Standard deviation per channel

print("=== Color Features ===")
print(f"Mean Intensity (R,G,B): {mean_color}")
print(f"Std Deviation (R,G,B): {std_color}\n")

# Step 4: Convert to Grayscale (2D matrix)
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

print("=== Grayscale Features ===")
print(f"Gray Image Shape: {gray.shape}")
print(f"Mean Intensity: {gray.mean():.2f}")
print(f"Standard Deviation: {gray.std():.2f}\n")

# Step 5: Blur the image (feature extraction: texture smoothness)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
print("Applied Gaussian Blur for feature extraction.\n")

# Step 6: Plot histograms for features
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 3)
colors = ('r', 'g', 'b')
for i, col in enumerate(colors):
    hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.title("RGB Color Histogram")

plt.subplot(2, 2, 4)
plt.hist(gray.ravel(), bins=256, color='gray')
plt.title("Grayscale Histogram")

plt.tight_layout()
plt.show()

# Step 7: (Optional) Write blurred image
cv2.imwrite("blurred_image.jpg", blurred)
print("Blurred image saved as 'blurred_image.jpg'.")
