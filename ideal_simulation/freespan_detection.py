from ideal_simulation.terrain_sonar_scann import *
from ideal_simulation.retriving_data_from_sonar import *
from ideal_simulation.circle_detection import *
from ideal_simulation.seafloor_detection import *

import numpy as np
import pyvista as pv
import os
def calculate_enclosed_area(curve_x, curve_y, xc, yc, radius):
    # Define the curve and circle geometries
    curve = LineString(np.column_stack([curve_x, curve_y]))
    circle = Point((xc, yc)).buffer(radius)
    intersection = curve.intersection(circle)
    
    if isinstance(intersection, LineString):
        # Points on the intersection line
        x_inter, y_inter = intersection.xy
        
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
        
        return enclosed_area, enclosed_percentage, polygon
    
    return 0, 0, None  # No significant intersection or error handling

def run_sonar_simulation_with_clustering(mesh_path, slice_position, dimensions, sonar_position, angles, max_range, angle_width, num_rays, clustering_params):
    terrain = pv.read(mesh_path)
    images_folder = "images/clustering_algorithms"
    os.makedirs(images_folder, exist_ok=True)

    slice_df = extract_2d_slice_from_mesh(terrain, slice_position, axis='x')
    if slice_df is not None:
        theta, distances = get_sonar_2d_plot(mesh_path, slice_position, dimensions, sonar_position, angles, max_range, angle_width, num_rays)
        x = -np.array(distances * np.sin(theta)) * 2  # Adjust this factor as needed
        y = np.array(distances * np.cos(theta))

        x_filtered, y_filtered = remove_most_common_y_value_with_margin(x, y)

        xc, yc, radius, common_mask = detect_circle(x_filtered, y_filtered, clustering_params)
        
        if xc is not None and yc is not None and radius is not None:
            print(f"Fitted Circle: Center = ({xc}, {yc}), Radius = {radius}")
            plot_and_save_points(x_filtered, y_filtered, common_mask, 'Common Circle Points', images_folder)
            plot_and_save_all_points_with_circle(x_filtered, y_filtered, common_mask, xc, yc, radius, images_folder)

            curve_x, curve_y = plot_curve_and_circle(x_filtered, y_filtered, common_mask, xc, yc, radius, images_folder)

            # Calculate intersection area and percentage
            enclosed_area, enclosed_percentage, enclosed_polygon = calculate_enclosed_area(curve_x, curve_y, xc, yc, radius)

            print(f"Enclosed Area: {enclosed_area}")
            print(f"Percentage Enclosed: {enclosed_percentage}%")

            if enclosed_polygon:
                plot_and_save_intersections(x_filtered, y_filtered, common_mask, curve_x, curve_y, xc, yc, radius, enclosed_polygon, images_folder)

            return np.column_stack((curve_x, curve_y)), np.column_stack((x_filtered, y_filtered)), enclosed_percentage
        else:
            print("No common points found among all clustering algorithms.")
    else:
        print("No data slice found for the given position.")