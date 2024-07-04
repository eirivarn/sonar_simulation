from ideal_simulation.terrain_sonar_scan import *
from ideal_simulation.circle_detection import *
from ideal_simulation.seafloor_detection import *
from ideal_simulation.terrain_sonar_scan import run_ideal_mesh_sonar_scan_simulation

import numpy as np
import pyvista as pv
import os
from shapely.geometry import LineString, Point, Polygon

def calculate_enclosed_area(curve_x, curve_y, xc, yc, radius):
    # Define the curve and circle geometries
    curve = LineString(np.column_stack([curve_x, curve_y]))
    circle = Point((xc, yc)).buffer(radius)
    intersection = curve.intersection(circle)
    bottom_of_circle_y = yc - radius

    # Calculate the distance from the bottom of the circle to the ocean floor
    ocean_floor_y = np.interp(xc, curve_x, curve_y)
    distance_to_ocean_floor = bottom_of_circle_y - ocean_floor_y
    relative_distance_to_ocean_floor = distance_to_ocean_floor / radius
    
    # Find the segment of the curve beneath the pipe
    x_min = xc - radius
    x_max = xc + radius
    segment_mask = (curve_x >= x_min) & (curve_x <= x_max)
    segment_x = curve_x[segment_mask]
    segment_y = curve_y[segment_mask]
    
    # Calculate the average slope of the segment beneath the pipe
    if len(segment_x) > 1:  # Ensure there are enough points to calculate a slope
        slopes = np.gradient(segment_y, segment_x)
        average_slope = np.mean(slopes)
    else:
        average_slope = None  # Not enough points to calculate a slope
    
    if isinstance(intersection, LineString):
        # Points on the intersection line
        x_inter, y_inter = intersection.xy
        
        if len(x_inter) == 0 or len(y_inter) == 0:
            return 0, 0, None, relative_distance_to_ocean_floor, average_slope
        
        # Calculate the slopes for the arc
        slopes_inter = np.gradient(y_inter, x_inter)
        
        # Generate points for the arc based on slopes
        arc_x = []
        arc_y = []
        for slope in slopes_inter:
            dx = radius / np.sqrt(1 + slope**2)
            dy = slope * dx
            arc_x.append(xc + dx)
            arc_y.append(yc + dy)
        
        # Closing the polygon
        polygon_points = np.vstack(([x_inter[-1], y_inter[-1]], np.column_stack([arc_x, arc_y]), [x_inter[0], y_inter[0]]))
        polygon = Polygon(polygon_points)
        
        enclosed_area = polygon.area
        circle_area = np.pi * radius ** 2
        enclosed_percentage = (enclosed_area / circle_area) * 100
        
        return enclosed_area, enclosed_percentage, polygon, relative_distance_to_ocean_floor, average_slope
    
    return 0, 0, None, relative_distance_to_ocean_floor, average_slope


def assess_pipe_condition(angle_degrees, enclosed_area, relative_distance_to_ocean_floor, radius, angle_weight=0.3, area_weight=0.3, distance_weight=0.4, free_span_threshold=0.1):
    # Default values in case of None
    if angle_degrees is None:
        angle_degrees = 0
    if enclosed_area is None:
        enclosed_area = 0
    if relative_distance_to_ocean_floor is None:
        relative_distance_to_ocean_floor = 0

    # Normalize the angle to a value between 0 and 1 (assuming max angle of 90 degrees is the most critical)
    normalized_angle = min(abs(angle_degrees) / 90.0, 1.0)
    
    # Normalize the enclosed area to a value between 0 and 1 (using the circle area for normalization)
    circle_area = np.pi * (radius ** 2)
    normalized_area = min(enclosed_area / circle_area, 1.0)
    
    # Use the relative distance directly
    normalized_distance = min(relative_distance_to_ocean_floor, 1.0)
    
    # Calculate stability score
    stability_score = (normalized_angle * angle_weight +
                       normalized_area * area_weight +
                       normalized_distance * distance_weight)
    
    # Invert the stability score to get the stability percentage
    stability_percentage = (1.0 - stability_score) * 100
    
    # Determine free-span status
    free_span_status = "Free-span" if relative_distance_to_ocean_floor > free_span_threshold else "Not in free-span"
    
    return free_span_status, stability_percentage

def run_pipeline_seafloor_detection(mesh_paths, slice_position, sonar_positions, angles, max_range, angle_width, num_rays, clustering_params):
    images_folder = "images/clustering_algorithms"
    os.makedirs(images_folder, exist_ok=True)

    signal_map, label_map = run_ideal_mesh_sonar_scan_simulation(mesh_paths, 'x', slice_position, sonar_positions, angles, max_range, angle_width, num_rays)

    # Ensure signal_map is a NumPy array
    signal_map = np.array(signal_map)

    x = signal_map[:, 0]
    y = signal_map[:, 1]
    
    x_circle, y_circle, radius, common_mask = detect_circle(x, y, clustering_params)
    if x_circle is not None and y_circle is not None and radius is not None:
        print(f"Fitted Circle: Center = ({x_circle}, {y_circle}), Radius = {radius}")
        plot_and_save_points(x, y, common_mask, 'Common Circle Points', images_folder)
        plot_and_save_all_points_with_circle(x, y, common_mask, x_circle, y_circle, radius, images_folder)

        curve_x, curve_y = plot_curve_and_circle(x, y, x_circle, y_circle, radius, images_folder)
         
        # Calculate intersection area and percentage
        enclosed_area, enclosed_percentage, enclosed_polygon, angle_degrees, relative_distance_to_ocean_floor = calculate_enclosed_area(curve_x, curve_y, x_circle, y_circle, radius)

        print(f"Enclosed Area: {enclosed_area}")
        print(f"Percentage Enclosed: {enclosed_percentage}%")
        print(f"Angle of seafloor under pipe: {angle_degrees} degrees")
        print(f"Relative distance to the size of the circle: {relative_distance_to_ocean_floor}")

        free_span_status, stability_percentage = assess_pipe_condition(angle_degrees, enclosed_area, relative_distance_to_ocean_floor, radius)
        print(f"Free-span Status: {free_span_status}")
        print(f"Stability Percentage: {stability_percentage}%")

        if enclosed_polygon:
            plot_and_save_intersections(x, y, common_mask, curve_x, curve_y, x_circle, y_circle, radius, enclosed_polygon, images_folder)

        return stability_percentage
    else:
        print("No common points found among all clustering algorithms.")

