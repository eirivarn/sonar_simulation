import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline

def gradient_orientation(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    return magnitude, angle

def filter_by_orientation(magnitude, angle, low_angle, high_angle):
    orientation_mask = (angle >= low_angle) & (angle <= high_angle)
    filtered_magnitude = np.zeros_like(magnitude)
    filtered_magnitude[orientation_mask] = magnitude[orientation_mask]
    return filtered_magnitude

def apply_hough_transform(image):
    if image.dtype != np.uint8:
        image = np.uint8(image)
    return cv2.HoughLinesP(image, 1, np.pi/180, threshold=10, minLineLength=50, maxLineGap=20)

def cluster_lines(lines):
    # Prepare data for clustering
    if lines is None:
        return None
    data = np.array([(x1, y1, x2, y2) for line in lines for x1, y1, x2, y2 in line])
    flattened_data = data.reshape(-1, 2)
    # Use DBSCAN to cluster data
    clustering = DBSCAN(eps=20, min_samples=2).fit(flattened_data)
    labels = clustering.labels_
    # Group lines by cluster labels
    grouped = {}
    for label, (x1, y1, x2, y2) in zip(labels, data):
        if label in grouped:
            grouped[label].append((x1, y1, x2, y2))
        else:
            grouped[label] = [(x1, y1, x2, y2)]
    return grouped

def fit_curve_to_clusters(grouped_lines):
    curves = []
    for label, lines in grouped_lines.items():
        all_points = []
        for line in lines:
            x1, y1, x2, y2 = line
            all_points.extend([(x1, y1), (x2, y2)])
        if len(all_points) < 10:
            continue  # Need at least 5 points to fit a reasonable spline
        all_points = np.array(all_points)
        # Filter out any extreme values or NaNs
        all_points = all_points[np.isfinite(all_points).all(axis=1)]
        # Sort points by x coordinate to fit a spline
        sorted_points = all_points[np.argsort(all_points[:, 0])]
        x, y = sorted_points[:, 0], sorted_points[:, 1]
        if len(x) < 5:  # Check again after filtering
            continue
        try:
            spline = UnivariateSpline(x, y, s=len(x))  # Adjust 's' if needed
            x_new = np.linspace(x.min(), x.max(), 20)  # More points for a smoother curve
            y_new = spline(x_new)
            if np.isnan(y_new).any():
                continue  # Skip this curve if NaNs are present
            curves.append((x_new, y_new))
        except Exception as e:
            print(f"Error fitting spline: {e}")
            continue
    return curves

# Load and process the image
image_path = 'sonar_results/position_-25.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image not found")
    exit()

magnitude, angle = gradient_orientation(image)
filtered = filter_by_orientation(magnitude, angle, 240, 360)
_, binary_image = cv2.threshold(filtered, 50, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilated = cv2.dilate(binary_image, kernel, iterations=3)
lines = apply_hough_transform(dilated)

if lines is not None:
    clustered_lines = cluster_lines(lines)
    curves = fit_curve_to_clusters(clustered_lines)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in curves:
        for i in range(len(x)-1):
            cv2.line(color_image, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), (0, 255, 0), 2)

cv2.imwrite('sonar_results/Curved_Seafloor_Estimation.png', color_image)
