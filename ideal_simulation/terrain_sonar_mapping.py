import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from ideal_simulation.terrain_sonar_scan import extract_2d_slice_from_mesh

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

def run_ideal_mesh_sonar_mapping_simulation(mesh_path, dimensions, axis, slice_positions, sonar_positions, angles, max_range, angle_width, num_rays):
    """ Run a terrain-based sonar simulation with given mesh and multiple sonar positions and visualize the results. """
    # Load and transform the mesh
    terrain = pv.read(mesh_path)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    terrain.points = terrain.points.dot(rotation_matrix)

    # Initialize list to store all sonar data
    all_sonar_hits = []

    # Iterate over slice positions and their configurations
    for slice_position in slice_positions:
        slice_position = -1 * slice_position

        # Extract 2D slice from the mesh
        slice_df = extract_2d_slice_from_mesh(terrain, slice_position, axis)
        if slice_df is not None:
            # Create binary map from slice data
            binary_map = create_binary_map(dimensions, slice_df)

            # Store sonar data for each sonar position and angle
            for sonar_position, sonar_angle in zip(sonar_positions, angles):
                sonar_hits = ray_cast(binary_map, sonar_position, sonar_angle, max_range, angle_width, num_rays)
                for hit in sonar_hits:
                    all_sonar_hits.append((slice_position, *hit))  
    
    # Visualization of results in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Unpack positions and coordinates for plotting
    y, z, x = zip(*all_sonar_hits)
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.set_box_aspect([10,10,1])

    ax.set_xlabel('X coordinate of sonar')
    ax.set_ylabel('Position along axis')
    ax.set_zlabel('Y coordinate of sonar')
    plt.show()
