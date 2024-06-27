from ideal_simulation.terrain_sonar_scann import *  # Assuming this imports necessary packages and functions
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

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

# Define the position and parameters for the sonar simulation
position = 11
dimensions = (1000, 1000)
sonar_position = (700, 500)
angle = 180
max_range = 700
angle_width = 45
num_rays = 50

# Ensure the directory exists
os.makedirs("sonar_results", exist_ok=True)
filename = f"sonar_results/position_{position}.png"

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
else:
    print("No slice data available to display.")
