from ideal_simulation.terrain_sonar_scann import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pyvista as pv

def create_sonar_image(sonar_data, max_range, num_rays):
    """ Create a sonar image from the ray cast data. """
    image = np.zeros((max_range, num_rays))  # Initialize image with zeros
    for i, (distance, strength) in enumerate(sonar_data):
        if 0 <= distance < max_range:
            image[distance, i] = strength

    # Apply smoothing (optional)
    image = cv2.GaussianBlur(image, (1, 1), 0)  # Adjust blur parameters as needed
    image = cv2.resize(image, (500, 400), interpolation=cv2.INTER_LINEAR)
    return image

def save_sonar_image(mesh_path, position, dimensions, sonar_position, angle, max_range, angle_width, num_rays, output_dir="images"):
    """ Save the sonar image from the terrain mesh data. """
    terrain = pv.read(mesh_path)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    terrain.points = terrain.points.dot(rotation_matrix)
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"simple_sonar_position_{position}.png")
    
    # Extract 2D slice from the mesh
    slice_df = extract_2d_slice_from_mesh(terrain, position, axis='x')
    
    if slice_df is not None:
        binary_map = create_binary_map_from_slice(dimensions, slice_df)
        
        # Perform ray-casting on the binary map
        sonar_data, theta = ray_cast(binary_map, sonar_position, angle, max_range, angle_width, num_rays)
        
        # Create sonar image from hits
        sonar_image = create_sonar_image(sonar_data, max_range, num_rays)  # Adjust max_range if necessary
        
        # Save the sonar image
        plt.imsave(filename, sonar_image)
        return filename
    else:
        print("No slice data available to display.")
        return None

def get_sonar_binary_map(mesh_path, position, dimensions, sonar_position, angle, max_range, angle_width, num_rays):
    """ Return the sonar data as a binary map. """
    terrain = pv.read(mesh_path)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    terrain.points = terrain.points.dot(rotation_matrix)
    
    # Extract 2D slice from the mesh
    slice_df = extract_2d_slice_from_mesh(terrain, position, axis='x')
    
    if slice_df is not None:
        binary_map = create_binary_map_from_slice(dimensions, slice_df)
        
        # Perform ray-casting on the binary map
        sonar_data, theta = ray_cast(binary_map, sonar_position, angle, max_range, angle_width, num_rays)
        return sonar_data, theta, binary_map
    else:
        print("No slice data available to display.")
        return None, None, None