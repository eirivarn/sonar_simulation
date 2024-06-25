import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.spatial import KDTree
from tqdm import tqdm

def bresenham_line(x0, y0, x1, y1):
    """Generate points using Bresenham's line algorithm."""
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
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

# Load the image for both pipeline and seafloor detection
image_path = 'sonar_results/position_-25.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Validate the image load
if image is None:
    raise ValueError("Image not loaded properly. Please check the file path.")

# Apply adaptive thresholding and invert for seafloor detection
binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
binary_image = cv2.bitwise_not(binary_image)  # Focus on dark areas

# Identify and connect original non-zero pixels
original_non_zero_points = np.column_stack(np.where(binary_image > 0))
radius = 5
kd_tree = KDTree(original_non_zero_points)
connected_image = binary_image.copy()

# Connect pixels using Bresenham's line algorithm
for i, (x0, y0) in enumerate(tqdm(original_non_zero_points, desc="Processing Pixels", ncols=100)):
    neighbors = kd_tree.query_ball_point([x0, y0], radius)
    for j in neighbors:
        if i != j:
            x1, y1 = original_non_zero_points[j]
            for (px, py) in bresenham_line(x0, y0, x1, y1):
                connected_image[px, py] = 255

# Label connected components
labeled_image, num_labels = label(connected_image, structure=np.ones((3, 3)))

# Find and save the largest connected component (seafloor)
max_size = 0
largest_component = None
for i in range(1, num_labels + 1):
    component_size = np.sum(labeled_image == i)
    if component_size > max_size:
        max_size = component_size
        largest_component = i

seafloor_mask = (labeled_image == largest_component).astype(np.uint8) * 255
cv2.imwrite('sonar_results/largest_seafloor_component.png', seafloor_mask)

# Detect pipelines using Hough Circles (Assuming sonar image is suited for this)
blurred = cv2.GaussianBlur(image, (9, 9), 0)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT, 2, 100,
                           param1=30, param2=0.1, minRadius=40, maxRadius=100)

pipeline_mask = np.zeros_like(image)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(pipeline_mask, (x, y), r, 255, -1)

# Calculate and display overlap
overlap_mask = cv2.bitwise_and(pipeline_mask, seafloor_mask)
overlap_area = np.sum(overlap_mask > 0)

# Calculate percentage overlap
pipeline_area = np.sum(pipeline_mask > 0)
overlap_percentage = (overlap_area / pipeline_area) * 100 if pipeline_area > 0 else 0

# Create a color composite image for better visualization
color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
color_image[seafloor_mask > 0] = [0, 255, 0]  # Green for seafloor
color_image[pipeline_mask > 0] = [255, 0, 0]  # Blue for pipeline
color_image[overlap_mask > 0] = [0, 0, 255]   # Red for overlap

# Display all results
fig, ax = plt.subplots(1, 5, figsize=(30, 6))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(seafloor_mask, cmap='gray')
ax[1].set_title('Seafloor Detected')
ax[2].imshow(pipeline_mask, cmap='gray')
ax[2].set_title('Pipeline Detected')
ax[3].imshow(overlap_mask, cmap='gray')
ax[3].set_title(f'Overlap (Area: {overlap_area} pixels)')
ax[4].imshow(color_image)
ax[4].set_title(f'Composite Image\nOverlap: {overlap_percentage:.2f}%')
plt.show()
