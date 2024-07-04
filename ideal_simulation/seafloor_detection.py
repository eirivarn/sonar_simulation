import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import os

def exclude_points_near_circle(x, y, xc, yc, radius, margin=25.0):
    distances = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    mask = (distances < (radius - margin)) | (distances > (radius + margin))
    return x[mask], y[mask]

def exclude_outliers(x, y, spline, threshold=5.0):
    residuals = y - spline(x)
    mask = np.abs(residuals) < threshold
    return x[mask], y[mask]

"""
Smoothing factor is a parameter that controls the trade-off between fitting the data and smoothing the curve
    A smaller value of smoothing factor will result in a curve that passes through more points
    A larger value of smoothing factor will result in a smoother curve that may not pass through all point

Outlier threshold is a parameter that controls the maximum residual value allowed for a point to be considered an outlier
    A smaller value of outlier threshold will result in more points being considered as outliers
    A larger value of outlier threshold will result in fewer points being considered as outliers
"""
def interpolate_remaining_points(x, y, smoothing_factor=10.0, outlier_threshold=0.5):

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    spline = UnivariateSpline(x_sorted, y_sorted, s=smoothing_factor)
    x_filtered, y_filtered = exclude_outliers(x_sorted, y_sorted, spline, outlier_threshold)
    
    # Re-fit the spline without outliers
    spline = UnivariateSpline(x_filtered, y_filtered, s=smoothing_factor)
    x_new = np.linspace(x_filtered.min(), x_filtered.max(), 500)
    y_new = spline(x_new)
    
    return x_new, y_new

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

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f'{title}.png')
    plt.savefig(file_path)
    plt.close()

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

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, 'All_Points_with_Fitted_Circle.png')
    plt.savefig(file_path)
    plt.close()

def plot_curve_and_circle(x, y, common_mask, xc, yc, radius, folder):
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

def calculate_enclosed_area_and_percentage(curve_x, curve_y, xc, yc, radius):
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

def plot_and_save_intersections(x, y, common_mask, curve_x, curve_y, xc, yc, radius, enclosed_polygon, folder):
    plt.figure(figsize=(15,15))
    plt.scatter(x, y, color='gray', label='All points')
    plt.scatter(x[common_mask], y[common_mask], color='red', label='Filtered points')
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
