import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from tqdm import tqdm
from scipy.spatial import KDTree

def bresenham_line(x0, y0, x1, y1):
    """Bresenham's line algorithm to generate points between two coordinates."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

# Load the sonar image
image_path = 'sonar_results/position_-25.png'  # Adjust the path as necessary
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded properly
if image is None:
    raise ValueError("Image not loaded properly. Please check the file path.")

# Apply adaptive thresholding to create a binary image
binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)

# Invert the binary image to consider dark areas as the foreground
binary_image = cv2.bitwise_not(binary_image)

# Identify original non-zero pixels
original_non_zero_points = np.column_stack(np.where(binary_image > 0))

# Define the radius within which to connect points
radius = 5  # Adjust the radius as needed

# Create a KD-Tree for the original non-zero points
kd_tree = KDTree(original_non_zero_points)

# Create an empty image to draw connections
connected_image = binary_image.copy()

# Connect original non-zero pixels within the radius using Bresenham's line algorithm
for i, (x0, y0) in enumerate(tqdm(original_non_zero_points, desc="Processing", ncols=100)):
    neighbors = kd_tree.query_ball_point([x0, y0], radius)
    for j in neighbors:
        if i == j:
            continue
        x1, y1 = original_non_zero_points[j]
        line_points = bresenham_line(x0, y0, x1, y1)
        for (px, py) in line_points:
            connected_image[px, py] = 255

# Ensure all original non-zero pixels are retained
connected_image[binary_image > 0] = 255

# Label connected components
labeled_image, num_labels = label(connected_image, structure=np.ones((3, 3)))

# Find the largest connected component
largest_component = None
max_size = 0

for i in range(1, num_labels + 1):
    component_size = np.sum(labeled_image == i)
    if component_size > max_size:
        max_size = component_size
        largest_component = i

# Create a mask for the largest connected component
largest_component_mask = (labeled_image == largest_component).astype(np.uint8) * 255

# Save the largest connected component
largest_component_image_path = 'largest_component.png'
cv2.imwrite(largest_component_image_path, largest_component_mask)

# Display the original, binary, connected, and largest component images
fig, ax = plt.subplots(1, 4, figsize=(24, 6))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(binary_image, cmap='gray')
ax[1].set_title('Binary Image')
ax[1].axis('off')
ax[2].imshow(connected_image, cmap='gray')
ax[2].set_title('Connected Pixels')
ax[2].axis('off')
ax[3].imshow(largest_component_mask, cmap='gray')
ax[3].set_title('Largest Connected Component')
ax[3].axis('off')
plt.show()
