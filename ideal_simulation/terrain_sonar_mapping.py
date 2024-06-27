import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from ideal_simulation.terrain_sonar_scann import extract_2d_slice_from_mesh, create_binary_map_from_slice

def ray_cast(room, pos, angle, max_range, angle_width, num_rays):
    """ Perform ray-casting to simulate sonar data and return hit coordinates. """
    rows, cols = room.shape
    sonar_hits = []
    
    for i in range(num_rays):
        ray_angle = angle - (angle_width / 2) + (angle_width * i / num_rays)
        ray_angle_rad = np.radians(ray_angle)

        for r in range(max_range):
            x = int(pos[0] + r * np.cos(ray_angle_rad))
            y = int(pos[1] + r * np.sin(ray_angle_rad))
            if x < 0 or x >= rows or y < 0 or y >= cols:
                break  # Stop when out of bounds
            if room[x, y] >= 0.5:
                sonar_hits.append((x, y))  # Add coordinates on hit
                break

    return sonar_hits

def run_ideal_mesh_sonar_mapping_simulation(mesh_path, dimensions, axis, sonar_positions, pos, angle, max_range, angle_width, num_rays):
    """ Run a terrain-based sonar simulation with given mesh and multiple sonar positions and visualize the results. """
    # Load and transform the mesh
    terrain = pv.read(mesh_path)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    terrain.points = terrain.points.dot(rotation_matrix)

    # Initialize list to store all sonar data
    all_sonar_data = []

    # Iterate over sonar positions and their configurations
    for idx, position in enumerate(sonar_positions):
        position = -position
        slice_df = extract_2d_slice_from_mesh(terrain, position, axis=axis)
        if slice_df is not None:
            binary_map = create_binary_map_from_slice((1000, 1000), slice_df)
            pos = (500, 500)  # Center sonar position on the map
            angle = angles[idx]
            sonar_data = ray_cast(binary_map, pos, angle, max_range, angle_width, num_rays)
            for data in sonar_data:
                all_sonar_data.append((position, *data))  # Save with position

    # Visualization of results in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Unpack positions and coordinates for plotting
    positions, ranges_angles, hit_coords = zip(*all_sonar_data)
    ranges, angles = zip(*ranges_angles)
    y, z, x = zip(*hit_coords)
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.set_box_aspect([10, 10, 1])

    ax.set_xlabel('X coordinate of sonar')
    ax.set_ylabel('Position along axis')
    ax.set_zlabel('Y coordinate of sonar')
    plt.show()
