from ideal_simulation.terrain_sonar_scann import *  
from ideal_simulation.retriving_data_from_sonar import *
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
import pyvista as pv

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

    ax.set_aspect('equal', adjustable='box')  # Ensure same scale on both axes
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Clustered Circle Points')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_sonar_simulation_with_clustering(mesh_path, slice_position, dimensions, sonar_position, angles, max_range, angle_width, num_rays, eps, min_samples):
    terrain = pv.read(mesh_path)

    # Ensure the directory exists
    os.makedirs("images", exist_ok=True)
    filename = f"images/position_{slice_position}.png"

    # Extract 2D slice from the mesh
    slice_df = extract_2d_slice_from_mesh(terrain, slice_position, axis='x')

    if slice_df is not None:
        theta, distances = get_sonar_2d_plot(mesh_path, slice_position, dimensions, sonar_position, angles, max_range, angle_width, num_rays)
        
        # Calculate x and y using polar to Cartesian conversion
        x = -np.array(distances * np.sin(theta))
        y = np.array(distances * np.cos(theta))
        
        # Cluster points and identify potential circle points
        circle_points, circle_mask = cluster_circle_points(x, y, eps, min_samples)
        plot_points(x, y, circle_mask)
    else:
        print("No data slice found for the given position.")