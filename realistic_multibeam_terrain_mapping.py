import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2

# Load and transform the mesh
terrain = pv.read('/Users/eirikvarnes/code/totalenergies/simulation_test/blender_terrain_test_1.obj')
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

def create_binary_map_from_slice(dimensions, slice_df):
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
            binary_map[x_index, y_index] = 0.65  # Default reflectivity for terrain material (rock/soil)
        else:
            print(f"Out of bounds: x_index={x_index}, y_index={y_index}")

    # Add debris and noise
    num_debris = random.randint(200, 500)
    for _ in range(num_debris):
        shape_type = random.choice(['circle', 'ellipse'])
        reflectivity = random.uniform(0.01, 0.1)  # Adjusting reflectivity for better detection
        if shape_type == 'circle':
            center = (random.randint(0, dimensions[1] - 1), random.randint(0, dimensions[0] - 1))
            radius = random.randint(1, 3)
            cv2.circle(binary_map, center, radius, reflectivity, -1)
        elif shape_type == 'ellipse':
            center = (random.randint(0, dimensions[1] - 1), random.randint(0, dimensions[0] - 1))
            axes = (random.randint(1, 3), random.randint(1, 2))
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
        return 0.01  # Moderate reflector (e.g., debris)
    else:
        return 0.001  # Weak reflector (e.g., sediment)

def calculate_multipath_reflections(material_value, incident_strength):
    """ Calculate reflections and transmissions based on material reflectivity. """
    reflectivity = material_reflectivity(material_value)
    reflected_strength = incident_strength * reflectivity
    transmitted_strength = incident_strength * (1 - reflectivity)
    return reflected_strength, transmitted_strength

def ray_cast(room, pos, angle, max_range, angle_width, num_rays, attenuation_factor=0.0003):
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

                # Continue propagating the ray
                if transmitted_strength < 0.1:  # Stop if the transmitted signal is too weak
                    break

        distorted_reflections = [(int(r + np.random.normal(0, 2)), strength * np.random.uniform(0.9, 1.1)) for r, strength in reflections]
        sonar_data.append(distorted_reflections if distorted_reflections else [(max_range, 0)])

    return sonar_data, theta

def transform_to_global(pos, sonar_data, theta):
    """ Transform intersections from sonar back to the global coordinate system. """
    global_coords = []
    for reflections, t in zip(sonar_data, theta):
        for (r, strength) in reflections:
            x = pos[0] + r * np.cos(t)
            y = pos[1] + r * np.sin(t)
            global_coords.append((x, y, strength))
    return global_coords

def transform_to_reference_sonar(ref_pos, ref_angle, global_coords):
    """ Transform global coordinates to the reference sonar's coordinate system. """
    transformed_coords = []
    ref_angle_rad = np.radians(ref_angle)
    cos_angle = np.cos(-ref_angle_rad)
    sin_angle = np.sin(-ref_angle_rad)
    for (x, y, strength) in global_coords:
        dx = x - ref_pos[0]
        dy = y - ref_pos[1]
        transformed_x = dx * cos_angle - dy * sin_angle
        transformed_y = dx * sin_angle + dy * cos_angle
        transformed_r = np.sqrt(transformed_x**2 + transformed_y**2)
        transformed_theta = np.arctan2(transformed_y, transformed_x)
        transformed_coords.append((transformed_r, transformed_theta, strength))
    return transformed_coords

def plot_both_views(room, sonar_positions, all_sonar_data, angles, angle_width, max_range, all_theta):
    """ Plot both room view and sonar image view as a cone in polar coordinates. """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for traditional room view
    ax1.imshow(room, cmap='turbo', origin='lower', interpolation='bilinear')
    colors = ['red', 'blue', 'green', 'yellow']
    
    for idx, (pos, sonar_data, theta) in enumerate(zip(sonar_positions, all_sonar_data, all_theta)):
        ax1.scatter([pos[1]], [pos[0]], color=colors[idx % len(colors)])  # Sonar position
        for reflections, t in zip(sonar_data, theta):
            for (r, strength) in reflections:
                x = pos[0] + r * np.cos(t)
                y = pos[1] + r * np.sin(t)
                ax1.plot([pos[1], y], [pos[0], x], 'r-', linewidth=0.5)
    
    ax1.set_title('Room with Pipe and Ground')

    # Create a 2D array to store signal strengths for the sonar view
    signal_grid = np.zeros((max_range, len(all_theta[0])))

    for i, (reflections, t) in enumerate(zip(all_sonar_data[0], all_theta[0])):
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

def main():
    dimensions = (1000, 1000)
    sonar_positions = [(500, 500), (500, 700), (500, 300)]
    angles = [180, 200, 150]
    max_range = 500
    angle_width = 45
    num_rays = 50

    positions = np.arange(-26, 0, 3)
    all_sonar_hits = []

    for position in positions:
        position = -position
        slice_df = extract_2d_slice_from_mesh(terrain, position, axis='x')
        if slice_df is not None:
            binary_map = create_binary_map_from_slice(dimensions, slice_df)

            all_sonar_data = []
            all_theta = []

            for sonar_position, angle in zip(sonar_positions, angles):
                sonar_data, theta = ray_cast(binary_map, sonar_position, angle, max_range, angle_width, num_rays)
                all_sonar_data.append(sonar_data)
                all_theta.append(theta)

                for reflections, t in zip(sonar_data, theta):
                    for r, strength in reflections:
                        if r < max_range:
                            x = int(sonar_position[0] + r * np.cos(t))
                            y = int(sonar_position[1] + r * np.sin(t))
                            all_sonar_hits.append((position, x, y))

    # Visualization of results in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Unpack positions and coordinates for plotting
    y, z, x = zip(*all_sonar_hits)
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=1)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X coordinate of sonar')
    ax.set_ylabel('Position along axis')
    ax.set_zlabel('Y coordinate of sonar')
    plt.show()

    # Plot last slice with sonar data
    if slice_df is not None:
        plot_both_views(binary_map, sonar_positions, all_sonar_data, angles, angle_width, max_range, all_theta)

if __name__ == "__main__":
    main()
