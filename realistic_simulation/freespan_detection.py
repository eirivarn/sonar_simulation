import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.spatial import KDTree
from tqdm import tqdm
from sklearn.decomposition import PCA

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
image_path = 'sonar_results/position_11.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Validate the image load
if image is None:
    raise ValueError("Image not loaded properly. Please check the file path.")

# Apply adaptive thresholding and invert for seafloor detection
binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
binary_image = cv2.bitwise_not(binary_image)  # Focus on dark areas

# Detect pipelines using Hough Circles
blurred = cv2.GaussianBlur(image, (9, 9), 0)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT, 2, 100,
                           param1=30, param2=0.1, minRadius=40, maxRadius=100)

pipeline_mask = np.zeros_like(image)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(pipeline_mask, (x, y), r, 255, -1)

# Find the top of the pipe
if circles is not None:
    top_of_pipe = min(circles[:, 1] - circles[:, 2])  # y - radius
else:
    top_of_pipe = 0  # If no pipe is detected, keep the entire image

# Remove everything above the top of the pipe
binary_image[:top_of_pipe, :] = 0

# Label connected components
labeled_image, num_labels = label(binary_image, structure=np.ones((3, 3)))

# Remove large objects (noise)
min_size = 500  # Adjust this value based on the expected size of noise objects
filtered_image = np.zeros_like(binary_image)

for i in range(1, num_labels + 1):
    component = (labeled_image == i)
    component_size = np.sum(component)
    if component_size < min_size:
        filtered_image[component] = 255

# Identify and connect original non-zero pixels
original_non_zero_points = np.column_stack(np.where(filtered_image > 0))

# Calculate PCA for the direction of maximum variance
pca = PCA(n_components=2)
pca.fit(original_non_zero_points)
principal_direction = pca.components_[0]

# Connect pixels using Bresenham's line algorithm biased towards principal direction
radius = 5
kd_tree = KDTree(original_non_zero_points)
connected_image = filtered_image.copy()

for i, (x0, y0) in enumerate(tqdm(original_non_zero_points, desc="Processing Pixels", ncols=100)):
    neighbors = kd_tree.query_ball_point([x0, y0], radius)
    for j in neighbors:
        if i != j:
            x1, y1 = original_non_zero_points[j]
            # Calculate the dot product to check alignment with principal direction
            vector = np.array([x1 - x0, y1 - y0])
            dot_product = np.dot(vector, principal_direction)
            if dot_product > 0:  # Connect only if aligned with principal direction
                for (px, py) in bresenham_line(x0, y0, x1, y1):
                    connected_image[px, py] = 255

# Label connected components again after connecting
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

# Calculate and draw the ellipse of variance
eclips_center = [200,200]
cov = np.cov(original_non_zero_points, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Get the largest eigenvalue and corresponding eigenvector
largest_eigenval = eigenvalues[-1]
largest_eigenvec = eigenvectors[:, -1]

# Get the smallest eigenvalue and corresponding eigenvector
smallest_eigenval = eigenvalues[0]
smallest_eigenvec = eigenvectors[:, 0]

# Angle of the ellipse
angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])

# Calculate width and height of the ellipse
width = 2 * np.sqrt(largest_eigenval)
height = 2 * np.sqrt(smallest_eigenval)

# Create the ellipse patch
ellipse = plt.matplotlib.patches.Ellipse(eclips_center, height, width, angle=np.degrees(angle), edgecolor='yellow', facecolor='none')

# Display all results
fig, ax = plt.subplots(1, 6, figsize=(36, 6))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(pipeline_mask, cmap='gray')
ax[1].set_title('Pipeline Detected')

# Add the ellipse to the plot
ax[2].imshow(filtered_image, cmap='gray')
ax[2].add_patch(ellipse)
ax[2].set_title('Ellipse of Variance')

ax[3].set_title('Seafloor Detected')
ax[3].imshow(seafloor_mask, cmap='gray')
ax[3].set_title('Pipeline Detected')
ax[4].imshow(overlap_mask, cmap='gray')
ax[4].set_title(f'Overlap (Area: {overlap_area} pixels)')
ax[5].imshow(color_image)
ax[5].set_title(f'Image\nOverlap: {overlap_percentage:.2f}%')
plt.show()