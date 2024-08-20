import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2
from noise import pnoise1

# Load and transform the mesh
terrain = pv.read('/Users/eirikvarnes/code/blender/combined_to_scale.obj')
rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
terrain.points = terrain.points.dot(rotation_matrix)

def extract_2d_slice_from_mesh(mesh, position, axis='x'):
    axes = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
    origins = {'x': (position, 0, 0), 'y': (0, position, 0), 'z': (0, 0, position)}
    normal = axes[axis]
    origin = origins[axis]

    slice = mesh.slice(normal=normal, origin=origin)

    if slice.n_points == 0:
        print(f"No points found in the slice at {axis}={position}")
        return None

    points = slice.points
    df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
    return df

def create_binary_map_from_slice(dimensions, slice_df, num_debris=100):
    """ Create a binary map from slice data with added realism and noise. """
    binary_map = np.zeros(dimensions)
    min_x, max_x = slice_df['Z'].min(), slice_df['Z'].max()
    min_y, max_y = slice_df['Y'].min(), slice_df['Y'].max()

    scale_x = (dimensions[0] / (max_x - min_x)) / 6
    scale_y = (dimensions[1] / (max_y - min_y))

    for _, row in slice_df.iterrows():
        x, y = row['Z'], row['Y']
        x_index = int((x - min_x) * scale_x + 800)
        y_index = int((y - min_y) * scale_y)
        x_index = dimensions[0] - 1 - x_index
        y_index = dimensions[1] - 1 - y_index

        if 0 <= x_index < dimensions[0] and 0 <= y_index < dimensions[1]:
            binary_map[x_index, y_index] = 0.75  # Default reflectivity for terrain material (rock/soil)
        else:
            print(f"Out of bounds: x_index={x_index}, y_index={y_index}")

    # Add debris and noise
    num_debris = num_debris
    for _ in range(num_debris):
        shape_type = random.choice(['circle', 'ellipse'])
        reflectivity = random.uniform(0.01, 0.05)  # Adjusting reflectivity for better detection
        if shape_type == 'circle':
            center = (random.randint(0, dimensions[1] - 1), random.randint(0, dimensions[0] - 1))
            radius = random.randint(1, 2)
            cv2.circle(binary_map, center, radius, reflectivity, -1)
        elif shape_type == 'ellipse':
            center = (random.randint(0, dimensions[1] - 1), random.randint(0, dimensions[0] - 1))
            axes = (random.randint(1, 2), random.randint(1, 2))
            angle = random.randint(0, 180)
            cv2.ellipse(binary_map, center, axes, angle, 0, 360, reflectivity, -1)

    # Apply Gaussian blur
    binary_map = cv2.GaussianBlur(binary_map, (5, 5), 0)

    return binary_map

def material_reflectivity(material_value):
    """ Determine reflectivity based on material value. """
    if material_value > 0.6:  # Adjusted threshold for debris
        return 0.75  # Strong reflector (e.g., metal)
    elif material_value > 0.3:
        return 0.001  # Moderate reflector (e.g., debris)
    else:
        return 0.0001  # Weak reflector (e.g., sediment)

def calculate_multipath_reflections(material_value, incident_strength):
    """ Calculate reflections and transmissions based on material reflectivity. """
    reflectivity = material_reflectivity(material_value)
    reflected_strength = incident_strength * reflectivity
    transmitted_strength = incident_strength * (1 - reflectivity)
    return reflected_strength, transmitted_strength

def ray_cast(room, pos, angle, max_range, angle_width, num_rays, attenuation_factor=0.0001):
    """ Perform ray-casting to simulate sonar data with multipath reflections and water attenuation. """
    rows, cols = room.shape
    sonar_data = []
    theta = []

    for i in range(num_rays):
        ray_angle = angle - (angle_width / 2) + (angle_width * i / num_rays)
        ray_angle_rad = np.radians(ray_angle)
        theta.append(ray_angle_rad)
        incident_strength = 1.0

        reflections = []
        for r in range(max_range):
            current_strength = incident_strength * np.exp(-attenuation_factor * r)
            x = int(pos[0] + r * np.cos(ray_angle_rad))
            y = int(pos[1] + r * np.sin(ray_angle_rad))
            if x < 0 or x >= rows or y < 0 or y >= cols:
                reflections.append((r, current_strength))
                break
            if room[x, y] > 0:  # Check for reflectivity instead of binary threshold
                reflected_strength, transmitted_strength = calculate_multipath_reflections(room[x, y], current_strength)
                reflections.append((r, reflected_strength))
                incident_strength = transmitted_strength
                # Debugging: Print the current status of the ray
                # print(f"Ray {i}: Hit at (x={x}, y={y}) with reflected_strength={reflected_strength}, transmitted_strength={transmitted_strength}")

                # Continue propagating the ray
                if transmitted_strength < 0.1:  # Stop if the transmitted signal is too weak
                    break

        distorted_reflections = [(int(r + np.random.normal(0, 2)), strength * np.random.uniform(0.9, 1.1)) for r, strength in reflections]
        sonar_data.append(distorted_reflections if distorted_reflections else [(max_range, 0)])

    return sonar_data, theta

def plot_both_views(room, sonar_position, sonar_data, angle, angle_width, max_range, theta):
    """ Plot both room view and sonar image view as a cone in polar coordinates. """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for traditional room view
    ax1.imshow(room, cmap='turbo', origin='lower', interpolation='bilinear')
    ax1.scatter([sonar_position[1]], [sonar_position[0]], color='red')  # Sonar position
    for reflections, t in zip(sonar_data, theta):
        for r, strength in reflections:
            x = sonar_position[0] + r * np.cos(t)
            y = sonar_position[1] + r * np.sin(t)
            ax1.plot([sonar_position[1], y], [sonar_position[0], x], 'r-', linewidth=0.5)

    # Create a 2D array to store signal strengths for the sonar view
    signal_grid = np.zeros((max_range, len(theta)))

    for i, (reflections, t) in enumerate(zip(sonar_data, theta)):
        for r, strength in reflections:
            if 0 <= r < max_range:
                signal_grid[r, i] = strength

    # Smooth the signal grid
    signal_grid = cv2.GaussianBlur(signal_grid, (5, 5), 0)
    
    # Plot for sonar image view as a cone
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_theta_zero_location('S')  # Set zero angle to the top (straight up)
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, max_range)
    ax2.set_xlim(-np.radians(angle_width / 2), np.radians(angle_width / 2))  # Center the sonar field of view
    ax2.set_title('Sonar Image')
    ax2.set_facecolor('white')

    # Use imshow to plot the signal grid
    extent = [-np.radians(angle_width / 2), np.radians(angle_width / 2), 0, max_range]
    ax2.imshow(signal_grid, aspect='auto', extent=extent, origin='lower', cmap='turbo', alpha=1)

    plt.show()


# Main Execution
dimensions = (2000, 2000)
sonar_position = (500, 500)
angle = 180
max_range = 500
angle_width = 45
num_rays = 50

# Extract a single slice and create binary map
position = 20
slice_df = extract_2d_slice_from_mesh(terrain, position, axis='x')

if slice_df is not None:
    binary_map = create_binary_map_from_slice(dimensions, slice_df)
    

    # Perform ray-casting on the binary map
    sonar_data, theta = ray_cast(binary_map, sonar_position, angle, max_range, angle_width, num_rays)

    # Debugging: Print the sonar data
    #for i, (reflections, t) in enumerate(zip(sonar_data, theta)):
    #    print(f"Ray {i}:")
    #    for r, strength in reflections:
    #        # print(f"  Distance={r}, Strength={strength}")

    # Visualize both views
    plot_both_views(binary_map, sonar_position, sonar_data, angle, angle_width, max_range, theta)
else:
    print("No slice data available to display.")
