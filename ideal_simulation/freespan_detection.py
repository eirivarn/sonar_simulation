from ideal_simulation.terrain_sonar_scann import *
from ideal_simulation.retriving_data_from_sonar import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pyvista as pv
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from skimage.measure import ransac
from scipy.optimize import least_squares


# Define a simple circle model for RANSAC
class CircleModel:
    def __init__(self):
        self.params = None

    def estimate(self, data):
        x_m = np.mean(data[:, 0])
        y_m = np.mean(data[:, 1])
        radii = np.sqrt((data[:, 0] - x_m) ** 2 + (data[:, 1] - y_m) ** 2)
        radius = np.mean(radii)
        self.params = (x_m, y_m, radius)
        return True

    def residuals(self, data):
        x_m, y_m, radius = self.params
        distances = np.sqrt((data[:, 0] - x_m) ** 2 + (data[:, 1] - y_m) ** 2)
        return np.abs(distances - radius)

    def predict_xy(self, t):
        x_m, y_m, radius = self.params
        return (x_m + radius * np.cos(t), y_m + radius * np.sin(t))
    

# Function to fit a circle using least squares optimization
def fit_circle_to_points(x, y):
    # Initial guess for the circle's center and radius
    x_m, y_m = np.mean(x), np.mean(y)
    r_initial = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))

    def residuals(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    result = least_squares(residuals, [x_m, y_m, r_initial], args=(x, y))
    xc, yc, radius = result.x
    return xc, yc, radius

def cluster_circle_points(x, y, algorithm='DBSCAN', **kwargs):
    points = np.column_stack((x, y))
    
    if algorithm == 'DBSCAN':
        clustering = DBSCAN(**kwargs).fit(points)
    elif algorithm == 'KMeans':
        clustering = KMeans(**kwargs).fit(points)
    elif algorithm == 'Agglomerative':
        clustering = AgglomerativeClustering(**kwargs).fit(points)
    elif algorithm == 'RANSAC':
        model, inliers = ransac(points, CircleModel, min_samples=3, residual_threshold=11, max_trials=1000)
        return points[inliers], inliers

    labels = clustering.labels_
    unique_labels = set(labels) - {-1} if -1 in labels else set(labels)

    if unique_labels:
        main_label = max(unique_labels, key=lambda label: (labels == label).sum())
        circle_points = points[labels == main_label]
        return circle_points, labels == main_label
    return np.array([]), np.zeros_like(labels, dtype=bool)

def plot_and_save_points(x, y, mask, title, folder):
    plt.figure()
    plt.scatter(x, y, color='gray', label='Non-circle points')
    if np.any(mask):
        plt.scatter(x[mask], y[mask], color='red', label='Circle points')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f'{title}.png')
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory

def plot_and_save_all_points_with_circle(x, y, common_mask, xc, yc, radius, folder):
    plt.figure()
    plt.scatter(x, y, color='gray', label='Non-circle points')
    plt.scatter(x[common_mask], y[common_mask], color='red', label='Common circle points')
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = xc + radius * np.cos(theta)
    y_circle = yc + radius * np.sin(theta)
    plt.plot(x_circle, y_circle, color='blue', label='Fitted circle')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('All Points with Fitted Circle')
    plt.legend()
    plt.grid(True)
    
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, 'All_Points_with_Fitted_Circle.png')
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory

def remove_most_common_y_value_with_margin(x, y, margin=2.0): #TODO Make a better system than hardcoded margin. Hard to tune clustering
    # Create a histogram of y values with a specified bin size (margin)
    hist, bin_edges = np.histogram(y, bins=np.arange(min(y), max(y) + margin, margin))
    # Find the bin with the highest frequency
    most_common_bin_index = np.argmax(hist)
    most_common_y = bin_edges[most_common_bin_index]
    # Create a mask to filter out points within the margin of the most common y value
    mask = (y < (most_common_y - margin / 2)) | (y > (most_common_y + margin / 2))
    return x[mask], y[mask]


def run_sonar_simulation_with_clustering(mesh_path, slice_position, dimensions, sonar_position, angles, max_range, angle_width, num_rays, clustering_params):
    terrain = pv.read(mesh_path)
    images_folder = "images/clustering_algorithms"
    os.makedirs(images_folder, exist_ok=True)

    slice_df = extract_2d_slice_from_mesh(terrain, slice_position, axis='x')
    if slice_df is not None:
        theta, distances = get_sonar_2d_plot(mesh_path, slice_position, dimensions, sonar_position, angles, max_range, angle_width, num_rays)
        x = -np.array(distances * np.sin(theta)) * 2  # Adjust for your needs
        y = np.array(distances * np.cos(theta))

        # Remove points with the most common rounded y value and return original coordinates
        x_filtered, y_filtered = remove_most_common_y_value_with_margin(x, y)

        
        titles = ['DBSCAN','RANSAC']
        circle_points_dict = {}

        for alg in titles:
            points, mask = cluster_circle_points(x_filtered, y_filtered, algorithm=alg, **clustering_params.get(alg, {}))
            circle_points_dict[alg] = points
            plot_and_save_points(x_filtered, y_filtered, mask, alg, images_folder)

        # Identify common points across all clustering algorithms
        all_masks = np.column_stack([np.isin(np.column_stack((x_filtered, y_filtered)), circle_points_dict[alg]).all(axis=1) for alg in titles])
        common_mask = all_masks.all(axis=1)
        common_points = np.column_stack((x_filtered, y_filtered))[common_mask]

        if len(common_points) > 0:
            xc, yc, radius = fit_circle_to_points(common_points[:, 0], common_points[:, 1])
            print(f"Fitted Circle: Center = ({xc}, {yc}), Radius = {radius}")
            plot_and_save_points(x_filtered, y_filtered, common_mask, 'Common Circle Points', images_folder)
            plot_and_save_all_points_with_circle(x_filtered, y_filtered, common_mask, xc, yc, radius, images_folder)
        else:
            print("No common points found among all clustering algorithms.")
    else:
        print("No data slice found for the given position.")

