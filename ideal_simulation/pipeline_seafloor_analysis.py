import numpy as np
import pyvista as pv
import os
from shapely.geometry import LineString, Point, Polygon
from ideal_simulation.terrain_sonar_scan import *
from ideal_simulation.circle_detection import *
from ideal_simulation.seafloor_detection import *
from ideal_simulation.terrain_sonar_scan import run_ideal_mesh_sonar_scan_simulation


def calculate_enclosed_area(curve_x, curve_y, xc, yc, radius):
    if len(curve_x) == 0 or len(curve_y) == 0:
        raise ValueError("Curve data is empty.")

    curve = LineString(np.column_stack([curve_x, curve_y]))
    circle = Point((xc, yc)).buffer(radius)
    intersection = curve.intersection(circle)
    bottom_of_circle_y = yc - radius

    if len(curve_x) < 2 or len(curve_y) < 2:
        raise ValueError("Not enough points in curve data to perform interpolation.")

    ocean_floor_y = np.interp(xc, curve_x, curve_y)
    distance_to_ocean_floor = bottom_of_circle_y - ocean_floor_y
    relative_distance_to_ocean_floor = distance_to_ocean_floor / radius
    
    x_min = xc - radius
    x_max = xc + radius
    segment_mask = (curve_x >= x_min) & (curve_x <= x_max)
    segment_x = curve_x[segment_mask]
    segment_y = curve_y[segment_mask]
    
    if len(segment_x) > 1:
        slopes = np.gradient(segment_y, segment_x)
        average_slope = abs(np.mean(slopes))
    else:
        average_slope = None
    
    if isinstance(intersection, LineString):
        x_inter, y_inter = intersection.xy
        
        if len(x_inter) == 0 or len(y_inter) == 0:
            return 0, 0, None, relative_distance_to_ocean_floor, average_slope
        
        start_angle = np.arctan2(y_inter[0] - yc, x_inter[0] - xc)
        end_angle = np.arctan2(y_inter[-1] - yc, x_inter[-1] - xc)
        
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle
        
        theta = np.linspace(start_angle, end_angle, 100)
        arc_x = xc + radius * np.cos(theta)
        arc_y = yc + radius * np.sin(theta)
        
        polygon_points = np.vstack(([x_inter[-1], y_inter[-1]], np.column_stack([arc_x, arc_y]), [x_inter[0], y_inter[0]]))
        polygon = Polygon(polygon_points)
        
        enclosed_area = polygon.area
        circle_area = np.pi * radius ** 2
        enclosed_percentage = (enclosed_area / circle_area) * 100
        
        return enclosed_area, enclosed_percentage, polygon, relative_distance_to_ocean_floor, average_slope
    
    return 0, 0, None, relative_distance_to_ocean_floor, average_slope

def assess_pipe_condition(angle_degrees, enclosed_area, relative_distance_to_ocean_floor, radius, angle_weight=0.3, area_weight=0.3, distance_weight=0.4, free_span_threshold=0.1):
    if angle_degrees is None:
        angle_degrees = 0
    if enclosed_area is None:
        enclosed_area = 0
    if relative_distance_to_ocean_floor is None:
        relative_distance_to_ocean_floor = 0

    normalized_angle = min(abs(angle_degrees) / 90.0, 1.0)
    
    circle_area = np.pi * (radius ** 2)
    normalized_area = (1 - min(enclosed_area / circle_area, 1.0))
    
    normalized_distance = min(relative_distance_to_ocean_floor, 1.0)
    
    stability_score = (normalized_angle * angle_weight +
                       normalized_area * area_weight +
                       normalized_distance * distance_weight)
    
    stability_percentage = (1.0 - stability_score) * 100
    
    free_span_status = "Free-span" if relative_distance_to_ocean_floor > free_span_threshold else "Not in free-span"
    
    return free_span_status, stability_percentage

def reduce_resolution_fast(x, y, num_bins=500):
    min_x, max_x = np.min(x), np.max(x)
    bins = np.linspace(min_x, max_x, num_bins)
    bin_indices = np.digitize(x, bins) - 1

    reduced_x = []
    reduced_y = []

    for i in range(num_bins):
        indices = np.where(bin_indices == i)[0]
        if len(indices) > 0:
            avg_x = np.mean(x[indices])
            avg_y = np.mean(y[indices])
            reduced_x.append(avg_x)
            reduced_y.append(avg_y)
    
    return np.array(reduced_x), np.array(reduced_y)

def extract_ground_truth(label_map, clustering_params):
    images_folder = "images/real"
    os.makedirs(images_folder, exist_ok=True)
    label_map_unique_values = np.unique(label_map)
    
    if label_map_unique_values.size > 1:
        circle_points = np.where(label_map == 2)
        curve_points = np.where(label_map == 1)
        
        if circle_points[0].size == 0 or curve_points[0].size == 0:
            print('No points found in the ground truth.')
            return None

        circle_x = circle_points[1]
        circle_y = circle_points[0]
        curve_x = curve_points[1]
        curve_y = curve_points[0]

        # Reduce resolution for curve points
        curve_x, curve_y = reduce_resolution_fast(curve_x, curve_y)

        x_circle, y_circle, radius, common_mask = detect_circle(circle_x, circle_y, clustering_params)
        if x_circle is None or y_circle is None or radius is None:
            print("GROUND TRUTH: Circle detection failed.")
            return None
        print(f"GROUND TRUTH: Fitted Circle: Center = ({x_circle}, {y_circle}), Radius = {radius}")
        plot_and_save_all_points_with_circle(circle_x, circle_y, common_mask, x_circle, y_circle, radius, images_folder)
        curve_x, curve_y = plot_curve_and_circle(curve_x, curve_y, x_circle, y_circle, radius, images_folder)
        
        enclosed_area, enclosed_percentage, enclosed_polygon, relative_distance_to_ocean_floor, angle_degrees = calculate_enclosed_area(curve_x, curve_y, x_circle, y_circle, radius)
        
        print(f"GROUND TRUTH: Enclosed Area: {enclosed_area}")
        print(f"GROUND TRUTH: Percentage Enclosed: {enclosed_percentage}%")
        print(f"GROUND TRUTH: Angle of seafloor under pipe: {angle_degrees} degrees")
        print(f"GROUND TRUTH: Relative distance to the size of the circle: {relative_distance_to_ocean_floor}")

        free_span_status, stability_percentage = assess_pipe_condition(angle_degrees, enclosed_area, relative_distance_to_ocean_floor, radius)
        print(f"GROUND TRUTH: Free-span Status: {free_span_status}")
        print(f"GROUND TRUTH: Stability Percentage: {stability_percentage}%")

        
        plot_and_save_intersections(circle_x, circle_y, common_mask, curve_x, curve_y, x_circle, y_circle, radius, enclosed_polygon, images_folder, map_type = 'real')

        return stability_percentage
        
    return None
        
def run_pipeline_seafloor_detection(mesh_paths, slice_position, sonar_positions, angles, max_range, angle_width, num_rays, clustering_params_signal, get_ground_truth=False, clustering_params_real=None):
    images_folder = "images/signal"
    os.makedirs(images_folder, exist_ok=True)

    signal_map, label_map = run_ideal_mesh_sonar_scan_simulation(mesh_paths, 'x', slice_position, sonar_positions, angles, max_range, angle_width, num_rays)

    signal_map = np.array(signal_map)

    x = signal_map[:, 0]
    y = signal_map[:, 1]
    
    x_circle, y_circle, radius, common_mask = detect_circle(x, y, clustering_params_signal)
    if x_circle is not None and y_circle is not None and radius is not None:
        print(f"SIGNAL: Fitted Circle: Center = ({x_circle}, {y_circle}), Radius = {radius}")
        plot_and_save_points(x, y, common_mask, 'Common Circle Points', images_folder)
        plot_and_save_all_points_with_circle(x, y, common_mask, x_circle, y_circle, radius, images_folder)

        curve_x, curve_y = plot_curve_and_circle(x, y, x_circle, y_circle, radius, images_folder)
         
        enclosed_area, enclosed_percentage, enclosed_polygon, relative_distance_to_ocean_floor, angle_degrees = calculate_enclosed_area(curve_x, curve_y, x_circle, y_circle, radius)

        print(f"SIGNAL: Enclosed Area: {enclosed_area}")
        print(f"SIGNAL: Percentage Enclosed: {enclosed_percentage}%")
        print(f"SIGNAL: Angle of seafloor under pipe: {angle_degrees} degrees")
        print(f"SIGNAL: Relative distance to the size of the circle: {relative_distance_to_ocean_floor}")

        free_span_status, stability_percentage = assess_pipe_condition(angle_degrees, enclosed_area, relative_distance_to_ocean_floor, radius)
        print(f"SIGNAL: Free-span Status: {free_span_status}")
        print(f"SIGNAL: Stability Percentage: {stability_percentage}%")

        plot_and_save_intersections(x, y, common_mask, curve_x, curve_y, x_circle, y_circle, radius, enclosed_polygon, images_folder)
        
        if get_ground_truth:
            ground_truth_stability_percentage = extract_ground_truth(label_map, clustering_params_real)
            return stability_percentage, ground_truth_stability_percentage

        return stability_percentage
    else:
        print("SIGNAL: No common points found among all clustering algorithms.")
