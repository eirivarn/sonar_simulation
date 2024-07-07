import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import ransac
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.optimize import least_squares
from typing import Tuple, List, Dict, Any, Union
from config import config

# Define CircleModel for RANSAC
class CircleModel:
    def __init__(self):
        self.params = None

    def estimate(self, data: np.ndarray) -> bool:
        x_m = np.mean(data[:, 0])
        y_m = np.mean(data[:, 1])
        radii = np.sqrt((data[:, 0] - x_m) ** 2 + (data[:, 1] - y_m) ** 2)
        radius = np.mean(radii)
        self.params = (x_m, y_m, radius)
        return True

    def residuals(self, data: np.ndarray) -> np.ndarray:
        x_m, y_m, radius = self.params
        distances = np.sqrt((data[:, 0] - x_m) ** 2 + (data[:, 1] - y_m) ** 2)
        return np.abs(distances - radius)

    def predict_xy(self, t: float) -> Tuple[float, float]:
        x_m, y_m, radius = self.params
        return (x_m + radius * np.cos(t), y_m + radius * np.sin(t))

# Define a function to fit a circle to points
def fit_circle_to_points(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x_m, y_m = np.mean(x), np.mean(y)
    r_initial = np.mean(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))

    def residuals(params: Tuple[float, float, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xc, yc, r = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    result = least_squares(residuals, [x_m, y_m, r_initial], args=(x, y))
    xc, yc, radius = result.x
    return xc, yc, radius

# Define a function to cluster circle points
def cluster_circle_points(x: np.ndarray, y: np.ndarray, algorithm: str = 'DBSCAN', is_real: bool = False, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    points = np.column_stack((x, y))

    if is_real:
        return points, np.ones(points.shape[0], dtype=bool)

    if algorithm == 'DBSCAN':
        clustering = DBSCAN(**kwargs).fit(points)
    elif algorithm == 'KMeans':
        clustering = KMeans(**kwargs).fit(points)
    elif algorithm == 'Agglomerative':
        clustering = AgglomerativeClustering(**kwargs).fit(points)
    elif algorithm == 'RANSAC':
        model, inliers = ransac(points, CircleModel, min_samples=config.get('clustering', 'ransac')['min_samples'], 
                                residual_threshold=config.get('clustering', 'ransac')['residual_threshold'], 
                                max_trials=config.get('clustering', 'ransac')['max_trials'])
        return points[inliers], inliers

    labels = clustering.labels_
    unique_labels = set(labels) - {-1} if -1 in labels else set(labels)

    if unique_labels:
        main_label = max(unique_labels, key=lambda label: (labels == label).sum())
        circle_points = points[labels == main_label]
        return circle_points, labels == main_label
    return np.array([]), np.zeros_like(labels, dtype=bool)

# Define a function to detect circles and save plots
def detect_circle(x: np.ndarray, y: np.ndarray, clustering_params: Dict[str, Dict[str, Union[int, float]]], algorithms: List[str] = ['KMeans', 'Agglomerative'], is_real: bool = False) -> Tuple[Union[float, None], Union[float, None], Union[float, None], np.ndarray]:
    circle_points_dict: Dict[str, np.ndarray] = {}

    for alg in algorithms:
        points, mask = cluster_circle_points(x, y, algorithm=alg, is_real=is_real, **clustering_params.get(alg, {}))
        circle_points_dict[alg] = points

        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, c='lightgray', label='All Points')

    if is_real:
        xc, yc, radius = fit_circle_to_points(x, y)
        return xc, yc, radius, np.ones(x.shape, dtype=bool)

    # Combine results for common points
    all_masks = np.column_stack([np.isin(np.column_stack((x, y)), circle_points_dict[alg]).all(axis=1) for alg in algorithms])
    common_mask = all_masks.all(axis=1)
    common_points = np.column_stack((x, y))[common_mask]

    if len(common_points) > 0:
        xc, yc, radius = fit_circle_to_points(common_points[:, 0], common_points[:, 1])
        return xc, yc, radius, common_mask
    else:
        return None, None, None, common_mask
