from typing import Tuple, List, Dict, Any, Union
import os
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from config import config
from ideal_simulation.terrain_sonar_scan import run_ideal_mesh_sonar_scan_simulation
from ideal_simulation.circle_detection import detect_circle
from ideal_simulation.seafloor_detection import plot_and_save_all_points_with_circle, plot_curve_and_circle, plot_and_save_points, plot_and_save_intersections

def calculate_enclosed_area(curve_x: np.ndarray, curve_y: np.ndarray, xc: float, yc: float, radius: float) -> Tuple[float, float, Union[Polygon, None], float, Union[float, None]]:
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

def assess_pipe_condition(angle_degrees: Union[float, None], enclosed_area: Union[float, None], relative_distance_to_ocean_floor: Union[float, None], radius: float) -> Tuple[str, float]:
    angle_weight = config.get('assessment', 'angle_weight')
    area_weight = config.get('assessment', 'area_weight')
    distance_weight = config.get('assessment', 'distance_weight')
    free_span_threshold = config.get('assessment', 'free_span_threshold')

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

def reduce_resolution_fast(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_bins = config.get('interpolation', 'num_bins')
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

def extract_ground_truth(label_map: np.ndarray, clustering_params: dict, is_real: bool = False) -> Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, float, float]]:
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

        x_circle, y_circle, radius, common_mask = detect_circle(circle_x, circle_y, clustering_params, is_real=is_real)
        if x_circle is None or y_circle is None or radius is None:
            print("GROUND TRUTH: Circle detection failed.")
            return None

        # Calculate the translation required to move (xc, yc) to (0, 0)
        translation_x = -x_circle
        translation_y = -y_circle

        # Apply the translation to all points
        circle_x_translated = circle_x + translation_x
        circle_y_translated = circle_y + translation_y
        curve_x_translated = curve_x + translation_x
        curve_y_translated = curve_y + translation_y
        x_circle_translated = x_circle + translation_x
        y_circle_translated = y_circle + translation_y
        
        print(f"GROUND TRUTH: Fitted Circle: Center = ({x_circle_translated}, {y_circle_translated}), Radius = {radius}")
        

        plot_and_save_all_points_with_circle(circle_x_translated, circle_y_translated, common_mask, x_circle_translated, y_circle_translated, radius, images_folder)
        curve_x_translated, curve_y_translated = plot_curve_and_circle(curve_x_translated, curve_y_translated, x_circle_translated, y_circle_translated, radius, images_folder)
        
        enclosed_area, enclosed_percentage, enclosed_polygon, relative_distance_to_ocean_floor, angle_degrees = calculate_enclosed_area(curve_x_translated, curve_y_translated, x_circle_translated, y_circle_translated, radius)
        
        print_assessment_results("GROUND TRUTH", enclosed_area, enclosed_percentage, angle_degrees, relative_distance_to_ocean_floor)

        free_span_status, stability_percentage = assess_pipe_condition(angle_degrees, enclosed_area, relative_distance_to_ocean_floor, radius)
        print(f"GROUND TRUTH: Free-span Status: {free_span_status}")
        print(f"GROUND TRUTH: Stability Percentage: {stability_percentage}%")

        plot_and_save_intersections(circle_x_translated, circle_y_translated, common_mask, curve_x_translated, curve_y_translated, x_circle_translated, y_circle_translated, radius, enclosed_polygon, images_folder, map_type='real')
        
        return x_circle_translated, y_circle_translated, radius, curve_x_translated, curve_y_translated, free_span_status, stability_percentage
    
    print('GROUND TRUTH: No unique values found in the label map.')
    return None

def print_assessment_results(prefix: str, enclosed_area: float, enclosed_percentage: float, angle_degrees: float, relative_distance_to_ocean_floor: float) -> None:
    print(f"{prefix}: Enclosed Area: {enclosed_area}")
    print(f"{prefix}: Percentage Enclosed: {enclosed_percentage}%")
    print(f"{prefix}: Angle of seafloor under pipe: {angle_degrees} degrees")
    print(f"{prefix}: Relative distance to the size of the circle: {relative_distance_to_ocean_floor}")

def run_pipeline_seafloor_detection(slice_position: int, 
                                    sonar_positions: List[Tuple[int, int]], 
                                    angles: List[int], 
                                    get_ground_truth: bool = False,
                                    ) -> Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, float, float, Union[None, Tuple[float, float, float, np.ndarray, np.ndarray]]]]:
    
    clustering_params = config.clustering_params

    images_folder = "images/signal"
    os.makedirs(images_folder, exist_ok=True)

    signal_map, label_map = run_ideal_mesh_sonar_scan_simulation(
        slice_position=slice_position,
        sonar_positions=sonar_positions, 
        angles=angles
    )

    signal_map = np.array(signal_map)

    x = signal_map[:, 0]
    y = signal_map[:, 1]
    
    x_circle, y_circle, radius, common_mask = detect_circle(x, y, clustering_params)
    if x_circle is not None and y_circle is not None and radius is not None:

        # Calculate the translation required to move (xc, yc) to (0, 0)
        translation_x = -x_circle
        translation_y = -y_circle

        # Apply the translation to all points
        x_translated = x + translation_x
        y_translated = y + translation_y
        x_circle_translated = x_circle + translation_x
        y_circle_translated = y_circle + translation_y
        
        print(f"SIGNAL: Fitted Circle: Center = ({x_circle_translated}, {y_circle_translated}), Radius = {radius}")

        plot_and_save_points(x_translated, y_translated, common_mask, 'Common Circle Points', images_folder)
        plot_and_save_all_points_with_circle(x_translated, y_translated, common_mask, x_circle_translated, y_circle_translated, radius, images_folder)

        curve_x, curve_y = plot_curve_and_circle(x_translated, y_translated, x_circle_translated, y_circle_translated, radius, images_folder)

        enclosed_area, enclosed_percentage, enclosed_polygon, relative_distance_to_ocean_floor, angle_degrees = calculate_enclosed_area(curve_x, curve_y, x_circle_translated, y_circle_translated, radius)
        print_assessment_results("SIGNAL", enclosed_area, enclosed_percentage, angle_degrees, relative_distance_to_ocean_floor)

        free_span_status, stability_percentage = assess_pipe_condition(angle_degrees, enclosed_area, relative_distance_to_ocean_floor, radius)
        print(f"SIGNAL: Free-span Status: {free_span_status}")
        print(f"SIGNAL: Stability Percentage: {stability_percentage}%")
        
        plot_and_save_intersections(x_translated, y_translated, common_mask, curve_x, curve_y, x_circle_translated, y_circle_translated, radius, enclosed_polygon, images_folder)
         
        if get_ground_truth:
            ground_truth_params = extract_ground_truth(label_map, clustering_params, is_real=True)
            return x_circle_translated, y_circle_translated, radius, curve_x, curve_y, free_span_status, stability_percentage, ground_truth_params

        return x_circle_translated, y_circle_translated, radius, curve_x, curve_y, free_span_status, stability_percentage
    else:
        print("SIGNAL: No common points found among all clustering algorithms.")
        return None