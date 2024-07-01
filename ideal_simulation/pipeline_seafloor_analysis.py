from ideal_simulation.terrain_sonar_scann import *
from ideal_simulation.retriving_data_from_sonar import *
from ideal_simulation.circle_detection import *
from ideal_simulation.seafloor_detection import *

import numpy as np
import pyvista as pv
import os
from shapely.geometry import LineString, Point, Polygon

def calculate_enclosed_area(curve_x, curve_y, xc, yc, radius):
    # Define the curve and circle geometries
    curve = LineString(np.column_stack([curve_x, curve_y]))
    circle = Point((xc, yc)).buffer(radius)
    intersection = curve.intersection(circle)
    
    if isinstance(intersection, LineString):
        # Points on the intersection line
        x_inter, y_inter = intersection.xy
        
        if len(x_inter) == 0 or len(y_inter) == 0:
            return 0, 0, None, None, None, None, None

        # Constructing a polygon from intersection and circle segment
        start_point = Point(x_inter[0], y_inter[0])
        end_point = Point(x_inter[-1], y_inter[-1])
        
        # Calculate the angles for the arc
        start_angle = np.arctan2(y_inter[0] - yc, x_inter[0] - xc)
        end_angle = np.arctan2(y_inter[-1] - yc, x_inter[-1] - xc)
        
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle
        
        # Generate points for the arc
        theta = np.linspace(start_angle, end_angle, 100)
        arc_x = xc + radius * np.cos(theta)
        arc_y = yc + radius * np.sin(theta)
        
        # Closing the polygon
        polygon_points = np.vstack(([x_inter[-1], y_inter[-1]], np.column_stack([arc_x, arc_y]), [x_inter[0], y_inter[0]]))
        polygon = Polygon(polygon_points)
        
        enclosed_area = polygon.area
        circle_area = np.pi * radius ** 2
        enclosed_percentage = (enclosed_area / circle_area) * 100
        
        # Calculate the angle between start_point and end_point
        delta_x = x_inter[-1] - x_inter[0]
        delta_y = y_inter[-1] - y_inter[0]
        angle_radians = np.arctan2(delta_y, delta_x)
        angle_degrees = np.degrees(angle_radians)
        print(f"Angle between start_point and end_point: {angle_degrees} degrees")
        
        # Calculate the distance from the bottom of the circle to the ocean floor
        bottom_of_circle_y = yc - radius
        ocean_floor_y = np.min(curve_y)
        distance_to_ocean_floor = bottom_of_circle_y - ocean_floor_y
        relative_distance_to_ocean_floor = distance_to_ocean_floor / radius
        print(f"Distance from the bottom of the circle to the ocean floor: {distance_to_ocean_floor}")
        print(f"Relative distance to the size of the circle: {relative_distance_to_ocean_floor}")

        return enclosed_area, enclosed_percentage, polygon, start_point, end_point, angle_degrees, relative_distance_to_ocean_floor
    
    return 0, 0, None, None, None, None, None  # No significant intersection or error handling

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

def run_pipeline_seafloor_detection(mesh_path, slice_position, dimensions, sonar_position, angles, max_range, angle_width, num_rays, clustering_params):
    terrain = pv.read(mesh_path)
    images_folder = "images/clustering_algorithms"
    os.makedirs(images_folder, exist_ok=True)

    slice_df = extract_2d_slice_from_mesh(terrain, slice_position, axis='x')
    if slice_df is not None:
        theta, distances = get_sonar_2d_plot(mesh_path, slice_position, dimensions, sonar_position, angles, max_range, angle_width, num_rays)
        x = -np.array(distances * np.sin(theta)) * 2  # Adjust this factor as needed
        y = np.array(distances * np.cos(theta))

        x_filtered, y_filtered = remove_most_common_y_value_with_margin(x, y)

        x_circle, y_circle, radius, common_mask = detect_circle(x_filtered, y_filtered, clustering_params)
        
        if x_circle is not None and y_circle is not None and radius is not None:
            print(f"Fitted Circle: Center = ({x_circle}, {y_circle}), Radius = {radius}")
            plot_and_save_points(x_filtered, y_filtered, common_mask, 'Common Circle Points', images_folder)
            plot_and_save_all_points_with_circle(x_filtered, y_filtered, common_mask, x_circle, y_circle, radius, images_folder)

            curve_x, curve_y = plot_curve_and_circle(x_filtered, y_filtered, common_mask, x_circle, y_circle, radius, images_folder)

            # Calculate intersection area and percentage
            enclosed_area, enclosed_percentage, enclosed_polygon, start_point, end_point, angle_degrees, relative_distance_to_ocean_floor = calculate_enclosed_area(curve_x, curve_y, x_circle, y_circle, radius)

            print(f"Enclosed Area: {enclosed_area}")
            print(f"Percentage Enclosed: {enclosed_percentage}%")
            print(f"Angle between start_point and end_point: {angle_degrees} degrees")
            print(f"Relative distance to the size of the circle: {relative_distance_to_ocean_floor}")

            free_span_status, stability_percentage = assess_pipe_condition(angle_degrees, enclosed_area, relative_distance_to_ocean_floor, radius)
            print(f"Free-span Status: {free_span_status}")
            print(f"Stability Percentage: {stability_percentage}%")

            if enclosed_polygon:
                plot_and_save_intersections(x_filtered, y_filtered, common_mask, curve_x, curve_y, x_circle, y_circle, radius, enclosed_polygon, images_folder)

            return np.column_stack((curve_x, curve_y)), np.column_stack((x_filtered, y_filtered)), enclosed_percentage
        else:
            print("No common points found among all clustering algorithms.")
    else:
        print("No data slice found for the given position.")
