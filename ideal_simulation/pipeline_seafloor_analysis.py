from typing import Tuple, List, Dict, Any, Union
import os
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from config import config
from ideal_simulation.terrain_sonar_scan import run_ideal_mesh_sonar_scan_simulation
from ideal_simulation.circle_detection import detect_circle
from ideal_simulation.seafloor_detection import (
    plot_and_save_all_points_with_circle,
    plot_curve_and_circle,
    plot_and_save_points,
    plot_and_save_intersections
)

def print_assessment_results(prefix: str, results: Dict[str, float]) -> None:
    for key, value in results.items():
        print(f"{prefix}: {key.replace('_', ' ').capitalize()}: {value}")

def print_status_and_stability(prefix: str, free_span_status: str, stability_percentage: float) -> None:
    print(f"{prefix}: Free-span Status: {free_span_status}")
    print(f"{prefix}: Stability Percentage: {stability_percentage}%")

def extract_curve_and_circle_points(label_map: Tuple[list, np.ndarray], map_type: str):
    if map_type == 'ground_truth':
        circle_points = np.where(label_map == 2)
        curve_points = np.where(label_map == 1)

        if circle_points[0].size == 0 or curve_points[0].size == 0:
            print('No points found in the ground truth.')
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        circle_x = circle_points[1]
        circle_y = circle_points[0]
        curve_x = curve_points[1]
        curve_y = curve_points[0]
    else: 
        circle_x = []
        circle_y = []
        curve_x = []
        curve_y = []
        for point in label_map:
            try:
                x, y, label = point  
                if label == 2:
                    circle_x.append(x)
                    circle_y.append(y)
                elif label == 1:
                    curve_x.append(x)
                    curve_y.append(y)
            except TypeError as e:
                print(f"Error processing point {point}: {e}")

    # Convert lists to numpy arrays for better performance in further processing
    circle_x = np.array(circle_x)
    circle_y = np.array(circle_y)
    curve_x = np.array(curve_x)
    curve_y = np.array(curve_y)

    return circle_x, circle_y, curve_x, curve_y


def reduce_resolution_fast(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_bins = config.get('interpolation', 'num_bins')
    bins = np.linspace(np.min(x), np.max(x), num_bins)
    bin_indices = np.digitize(x, bins) - 1

    reduced_x, reduced_y = [], []
    for i in range(num_bins):
        indices = np.where(bin_indices == i)[0]
        if indices.size > 0:
            reduced_x.append(np.mean(x[indices]))
            reduced_y.append(np.mean(y[indices]))
    
    return np.array(reduced_x), np.array(reduced_y)

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
    
    segment_mask = (curve_x >= xc - radius) & (curve_x <= xc + radius)
    segment_x, segment_y = curve_x[segment_mask], curve_y[segment_mask]
    
    average_slope = np.mean(np.abs(np.gradient(segment_y, segment_x))) if len(segment_x) > 1 else None
    
    if isinstance(intersection, LineString):
        return _calculate_polygon_metrics(intersection, xc, yc, radius, relative_distance_to_ocean_floor, average_slope)
    
    return 0, 0, None, relative_distance_to_ocean_floor, average_slope

def _calculate_polygon_metrics(intersection: LineString, xc: float, yc: float, radius: float, relative_distance_to_ocean_floor: float, average_slope: float) -> Tuple[float, float, Union[Polygon, None], float, Union[float, None]]:
    x_inter, y_inter = intersection.xy
    if len(x_inter) == 0 or len(y_inter) == 0:
        return 0, 0, None, relative_distance_to_ocean_floor, average_slope

    start_angle, end_angle = sorted([np.arctan2(y_inter[i] - yc, x_inter[i] - xc) for i in [0, -1]])
    theta = np.linspace(start_angle, end_angle, 100)
    arc_x = xc + radius * np.cos(theta)
    arc_y = yc + radius * np.sin(theta)

    polygon_points = np.vstack(([x_inter[-1], y_inter[-1]], np.column_stack([arc_x, arc_y]), [x_inter[0], y_inter[0]]))
    polygon = Polygon(polygon_points)

    enclosed_area = polygon.area
    circle_area = np.pi * radius ** 2
    enclosed_percentage = (enclosed_area / circle_area) * 100

    return enclosed_area, enclosed_percentage, polygon, relative_distance_to_ocean_floor, average_slope

def assess_pipe_condition(angle_degrees: Union[float, None], enclosed_area: Union[float, None], relative_distance_to_ocean_floor: Union[float, None], radius: float) -> Tuple[str, float]:
    angle_weight = config.get('assessment', 'angle_weight')
    area_weight = config.get('assessment', 'area_weight')
    distance_weight = config.get('assessment', 'distance_weight')
    free_span_threshold = config.get('assessment', 'free_span_threshold')

    normalized_angle = min(abs(angle_degrees or 0) / 90.0, 1.0)
    normalized_area = 1 - min((enclosed_area or 0) / (np.pi * radius ** 2), 1.0)
    normalized_distance = min(relative_distance_to_ocean_floor or 0, 1.0)
    
    stability_score = (normalized_angle * angle_weight +
                       normalized_area * area_weight +
                       normalized_distance * distance_weight)
    
    stability_percentage = (1.0 - stability_score) * 100
    free_span_status = "Free-span" if relative_distance_to_ocean_floor > free_span_threshold else "Not in free-span"
    
    return free_span_status, stability_percentage

def extract_ground_truth(label_map: np.ndarray, clustering_params: dict, is_real: bool = False) -> Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, float, float]]:
    images_folder = "images/real"
    os.makedirs(images_folder, exist_ok=True)
    
    label_map_unique_values = np.unique(label_map)
    if label_map_unique_values.size <= 1:
        print('GROUND TRUTH: No unique values found in the label map.')
        return None

    circle_x, circle_y, curve_x, curve_y = extract_curve_and_circle_points(label_map, 'ground_truth')
    if circle_x.size == 0 or curve_x.size == 0:
        return None

    curve_x, curve_y = reduce_resolution_fast(curve_x, curve_y)

    x_circle, y_circle, radius, common_mask = detect_circle(circle_x, circle_y, clustering_params, is_real=is_real)
    if x_circle is None or y_circle is None or radius is None:
        print("GROUND TRUTH: Circle detection failed.")
        return None

    # Apply the translation to all points
    translation_x, translation_y = -x_circle, -y_circle
    circle_x_translated, circle_y_translated = circle_x + translation_x, circle_y + translation_y
    curve_x_translated, curve_y_translated = curve_x + translation_x, curve_y + translation_y
    x_circle_translated, y_circle_translated = x_circle + translation_x, y_circle + translation_y

    print(f"GROUND TRUTH: Fitted Circle: Center = ({x_circle_translated}, {y_circle_translated}), Radius = {radius}")

    plot_and_save_all_points_with_circle(circle_x_translated, circle_y_translated, common_mask, x_circle_translated, y_circle_translated, radius, images_folder)
    curve_x_translated, curve_y_translated = plot_curve_and_circle(curve_x_translated, curve_y_translated, x_circle_translated, y_circle_translated, radius, images_folder)

    enclosed_area, enclosed_percentage, enclosed_polygon, relative_distance_to_ocean_floor, angle_degrees = calculate_enclosed_area(curve_x_translated, curve_y_translated, x_circle_translated, y_circle_translated, radius)
    
    results = {
        'enclosed_area': enclosed_area,
        'enclosed_percentage': enclosed_percentage,
        'angle_degrees': angle_degrees,
        'relative_distance_to_ocean_floor': relative_distance_to_ocean_floor
    }
    print_assessment_results("GROUND TRUTH", results)

    free_span_status, stability_percentage = assess_pipe_condition(angle_degrees, enclosed_area, relative_distance_to_ocean_floor, radius)
    print_status_and_stability("GROUND TRUTH", free_span_status, stability_percentage)

    plot_and_save_intersections(circle_x_translated, circle_y_translated, common_mask, curve_x_translated, curve_y_translated, x_circle_translated, y_circle_translated, radius, enclosed_polygon, images_folder, map_type='real')
    
    return x_circle_translated, y_circle_translated, radius, curve_x_translated, curve_y_translated, free_span_status, stability_percentage, enclosed_percentage, relative_distance_to_ocean_floor, angle_degrees

def run_pipeline_seafloor_detection(slice_position: int, 
                                    sonar_positions: List[Tuple[int, int]], 
                                    angles: List[int], 
                                    get_ground_truth: bool = False,
                                    use_clustering: bool = False
                                    ) -> Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, float, float, Union[None, Tuple[float, float, float, np.ndarray, np.ndarray]]]]:
    
    clustering_params = config.clustering_params

    images_folder = "images/signal"
    os.makedirs(images_folder, exist_ok=True)

    signal_map, label_map = run_ideal_mesh_sonar_scan_simulation(
        slice_position=slice_position,
        sonar_positions=sonar_positions, 
        angles=angles
    )

    x, y = np.array(signal_map)[:, 0], np.array(signal_map)[:, 1]
    
    if use_clustering:
        x_circle, y_circle, radius, common_mask = detect_circle(x, y, clustering_params)
        if x_circle is None or y_circle is None or radius is None:
            print("SIGNAL: No common points found among all clustering algorithms.")
            return None
        plot_and_save_points(x_translated, y_translated, common_mask, 'Common Circle Points', images_folder)
        plot_and_save_all_points_with_circle(x_translated, y_translated, common_mask, x_circle_translated, y_circle_translated, radius, images_folder)

    else: 
        circle_x, circle_y, _, _ = extract_curve_and_circle_points(signal_map, 'signal')
        x_circle, y_circle, radius, common_mask = detect_circle(circle_x, circle_y, clustering_params, is_real=True)

    # Apply the translation to all points
    translation_x, translation_y = -x_circle, -y_circle
    x_translated, y_translated = x + translation_x, y + translation_y
    circle_x_translated, circle_y_translated = circle_x + translation_x, circle_y + translation_y
    x_circle_translated, y_circle_translated = x_circle + translation_x, y_circle + translation_y
    
    print(f"SIGNAL: Fitted Circle: Center = ({x_circle_translated}, {y_circle_translated}), Radius = {radius}")

    
    curve_x, curve_y = plot_curve_and_circle(x_translated, y_translated, x_circle_translated, y_circle_translated, radius, images_folder)

    enclosed_area, enclosed_percentage, enclosed_polygon, relative_distance_to_ocean_floor, angle_degrees = calculate_enclosed_area(curve_x, curve_y, x_circle_translated, y_circle_translated, radius)
    
    results = {
        'enclosed_area': enclosed_area,
        'enclosed_percentage': enclosed_percentage,
        'angle_degrees': angle_degrees,
        'relative_distance_to_ocean_floor': relative_distance_to_ocean_floor
    }
    print_assessment_results("SIGNAL", results)

    free_span_status, stability_percentage = assess_pipe_condition(angle_degrees, enclosed_area, relative_distance_to_ocean_floor, radius)
    print_status_and_stability("SIGNAL", free_span_status, stability_percentage)

    plot_and_save_intersections(circle_x_translated, circle_y_translated, common_mask[:x_translated.size], curve_x, curve_y, x_circle_translated, y_circle_translated, radius, enclosed_polygon, images_folder)
     
    if get_ground_truth:
        ground_truth_params = extract_ground_truth(label_map, clustering_params, is_real=True)
        return x_circle_translated, y_circle_translated, radius, curve_x, curve_y, free_span_status, stability_percentage, enclosed_percentage, relative_distance_to_ocean_floor, angle_degrees, ground_truth_params

    return x_circle_translated, y_circle_translated, radius, curve_x, curve_y, free_span_status, stability_percentage, enclosed_percentage, relative_distance_to_ocean_floor, angle_degrees