import numpy as np
from skimage.measure import ransac
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.optimize import least_squares

#TODO Tune circle algorithms 

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

def fit_circle_to_points(x, y):
    x_m, y_m = np.mean(x), np.mean(y)
    r_initial = np.mean(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))

    def residuals(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

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
        model, inliers = ransac(points, CircleModel, min_samples=20, residual_threshold=50, max_trials=1000)
        return points[inliers], inliers

    labels = clustering.labels_
    unique_labels = set(labels) - {-1} if -1 in labels else set(labels)

    if unique_labels:
        main_label = max(unique_labels, key=lambda label: (labels == label).sum())
        circle_points = points[labels == main_label]
        return circle_points, labels == main_label
    return np.array([]), np.zeros_like(labels, dtype=bool)

def detect_circle(x, y, clustering_params):
    titles = ['DBSCAN', 'RANSAC']
    circle_points_dict = {}

    for alg in titles:
        points, mask = cluster_circle_points(x, y, algorithm=alg, **clustering_params.get(alg, {}))
        circle_points_dict[alg] = points

    all_masks = np.column_stack([np.isin(np.column_stack((x, y)), circle_points_dict[alg]).all(axis=1) for alg in titles])
    common_mask = all_masks.all(axis=1)
    common_points = np.column_stack((x, y))[common_mask]

    if len(common_points) > 0:
        xc, yc, radius = fit_circle_to_points(common_points[:, 0], common_points[:, 1])
        return xc, yc, radius, common_mask
    else:
        return None, None, None, common_mask
