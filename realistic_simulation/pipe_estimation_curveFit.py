from ideal_simulation.terrain_sonar_scan import *  # Assuming this imports necessary packages and functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import cv2
def extract_points_from_binary_map(binary_map):
    points = np.argwhere(binary_map == 1)
    # Convert from matrix indices to coordinates
    x = points[:, 1]
    y = binary_map.shape[0] - points[:, 0]
    return x, y

def cluster_circle_points(x, y, eps, min_samples):
    points = np.column_stack((x, y))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # Identify cluster labels (ignore noise points labeled as -1)
    unique_labels = set(labels) - {-1}

    # Assuming the largest cluster by count that is not noise is the circle
    if unique_labels:
        main_label = max(unique_labels, key=lambda label: (labels == label).sum())
        circle_points = points[labels == main_label]
        return circle_points, labels == main_label
    return np.array([]), np.zeros_like(labels, dtype=bool)  # No valid circle found

def plot_points(x, y, circle_mask):
    fig, ax = plt.subplots()
    # Plot all points in gray
    ax.scatter(x, y, color='gray', label='Non-circle points')
    # Highlight circle points in red
    if np.any(circle_mask):
        ax.scatter(x[circle_mask], y[circle_mask], color='red', label='Circle points')
    ax.set_aspect('equal', adjustable='datalim')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Clustered Circle Points')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define the position and parameters for the sonar simulation
position = 11
dimensions = (1000, 1000)
sonar_position = (700, 500)
angle = 180
max_range = 700
angle_width = 45
num_rays = 50

slice_df = extract_2d_slice_from_mesh(terrain, position, axis='x')

if slice_df is not None:
    binary_map = create_binary_map_from_slice(dimensions, slice_df)

    # Extract (x, y) points from the binary map
    x, y = extract_points_from_binary_map(binary_map)

    # Cluster points and identify potential circle points
    circle_points, circle_mask = cluster_circle_points(x, y, eps=1.5, min_samples=3)

    # Plot the results
    plot_points(x, y, circle_mask)    
    
    sonar_data, theta = ray_cast(binary_map, sonar_position, angle, max_range, angle_width, num_rays)
    
    
else:
    print("No data slice found for the given position.")
    
def extract_points_from_image(image):
    """Extract non-zero points from an image."""
    points = np.argwhere(image > 0)  # Get the coordinates of non-zero points
    return points[:, 1], points[:, 0]  # Return x, y coordinates

def plot_clusters(data, labels):
    """Plot data points with different clusters."""
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    
    plt.title('Sonar Data Clustering')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.grid(True)
    plt.show()

image = cv2.imread('sonar_results/position_11.png', cv2.IMREAD_GRAYSCALE)
binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
binary_image = cv2.bitwise_not(binary_image)

# Extract (x, y) points from the image
x, y = extract_points_from_image(binary_image)
points = np.column_stack((x, -y))

# DBSCAN clustering
db = DBSCAN(eps=30, min_samples=60).fit(points)
labels = db.labels_

# Plot the clusters
plot_clusters(points, labels)