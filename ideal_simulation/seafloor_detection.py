import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import os
from config import config
from typing import List, Tuple, Union

def exclude_points_near_circle(x: np.ndarray, y: np.ndarray, xc: float, yc: float, radius: float, margin: float = None) -> Tuple[np.ndarray, np.ndarray]:
    if margin is None:
        margin = config.get('interpolation', 'circle_point_margin')
    distances = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    mask = (distances < (radius - margin)) | (distances > (radius + margin))
    return x[mask], y[mask]

def exclude_outliers(x: np.ndarray, y: np.ndarray, threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
    if threshold is None:
        threshold = config.get('interpolation', 'curve_outlier_threshold')
    # In the case of linear interpolation, we assume outliers are defined by some external criteria, here is a placeholder.
    # Implement your outlier detection mechanism if needed.
    mask = np.abs(y - np.mean(y)) < threshold  # Example placeholder
    return x[mask], y[mask]

def interpolate_remaining_points(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Linear interpolation between points
    x_new = np.linspace(x_sorted.min(), x_sorted.max(), 500)
    y_new = np.interp(x_new, x_sorted, y_sorted)

    return x_new, y_new

def plot_and_save_points(x: np.ndarray, y: np.ndarray, mask: np.ndarray, title: str, folder: str) -> None:
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

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f'{title}.png')
    plt.savefig(file_path)
    plt.close()

def plot_and_save_all_points_with_circle(x: np.ndarray, y: np.ndarray, common_mask: np.ndarray, xc: float, yc: float, radius: float, folder: str) -> None:
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

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, 'All_Points_with_Fitted_Circle.png')
    plt.savefig(file_path)
    plt.close()

def plot_curve_and_circle(x: np.ndarray, y: np.ndarray, xc: float, yc: float, radius: float, folder: str) -> Tuple[np.ndarray, np.ndarray]:
    x_remaining, y_remaining = exclude_points_near_circle(x, y, xc, yc, radius)
    x_new, y_new = interpolate_remaining_points(x_remaining, y_remaining)

    plt.figure()
    plt.scatter(x, y, color='gray', label='Original points')
    plt.plot(x_new, y_new, color='green', label='Interpolated curve')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Interpolated Curve from Remaining Points')
    plt.legend()
    plt.grid(True)

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, 'Interpolated_Curve.png')
    plt.savefig(file_path)
    plt.close()

    return x_new, y_new

def calculate_enclosed_area_and_percentage(curve_x: np.ndarray, curve_y: np.ndarray, xc: float, yc: float, radius: float) -> Tuple[float, float, Union[Polygon, None]]:
    curve = LineString(np.column_stack([curve_x, curve_y]))
    circle = Point(xc, yc).buffer(radius)
    intersection = curve.intersection(circle)

    if isinstance(intersection, LineString):
        # Use the endpoints of the LineString to form a polygon with the circle segment
        x_inter, y_inter = intersection.xy
        start_angle = np.arctan2(y_inter[0] - yc, x_inter[0] - xc)
        end_angle = np.arctan2(y_inter[-1] - yc, x_inter[-1] - xc)

        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle

        theta = np.linspace(start_angle, end_angle, 100)
        arc_x = xc + radius * np.cos(theta)
        arc_y = yc + radius * np.sin(theta)

        polygon_points = np.vstack((np.column_stack([x_inter, y_inter]), np.column_stack([arc_x[::-1], arc_y[::-1]])))
        enclosed_polygon = Polygon(polygon_points)
        enclosed_area = enclosed_polygon.area
    elif isinstance(intersection, Polygon):
        enclosed_polygon = intersection
        enclosed_area = intersection.area
    else:
        enclosed_polygon = None
        enclosed_area = 0.0

    circle_area = np.pi * radius ** 2
    enclosed_percentage = (enclosed_area / circle_area) * 100 if circle_area > 0 else 0

    return enclosed_area, enclosed_percentage, enclosed_polygon

def plot_and_save_intersections(x: np.ndarray, y: np.ndarray, common_mask: np.ndarray, curve_x: np.ndarray, curve_y: np.ndarray, xc: float, yc: float, radius: float, enclosed_polygon: Union[Polygon, None], folder: str, map_type: str = 'signal') -> None:
    plt.figure(figsize=(15, 15))
    if map_type == 'signal':
        plt.scatter(x[common_mask], y[common_mask], color='red', label='Filtered points')
        plt.scatter(x, y, color='gray', label='All points')
    plt.plot(curve_x, curve_y, color='green', label='Interpolated curve')
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = xc + radius * np.cos(theta)
    y_circle = yc + radius * np.sin(theta)
    plt.plot(x_circle, y_circle, color='blue', label='Fitted circle')

    if enclosed_polygon:
        x_enclosed, y_enclosed = enclosed_polygon.exterior.xy
        plt.fill(x_enclosed, y_enclosed, color='yellow', alpha=0.5, label='Enclosed Area')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Intersection and Enclosed Area')
    plt.legend()
    plt.grid(True)

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, 'Intersection_Enclosed_Area.png')
    plt.savefig(file_path)
    plt.close()
