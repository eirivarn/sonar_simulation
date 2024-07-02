import pyvista as pv
import numpy as np
import pandas as pd
from ideal_simulation.multiple_sonar import ray_cast, plot_both_views
import matplotlib.pyplot as plt 

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

def create_binary_map(df, grid_size, x_range, y_range):
    if df is None:
        return None

    # Extend the range slightly to ensure all data points are included in the bins
    margin_x = (x_range[1] - x_range[0]) * 0.01  # 1% of the range as a margin
    margin_y = (y_range[1] - y_range[0]) * 0.01  # 1% of the range as a margin

    x_bins = np.linspace(x_range[0] - margin_x, x_range[1] + margin_x, grid_size[0] + 1)
    y_bins = np.linspace(y_range[0] - margin_y, y_range[1] + margin_y, grid_size[1] + 1)

    binary_map, _, _ = np.histogram2d(df['Y'], df['Z'], bins=[x_bins, y_bins])

    # Ensure that binary_map has the exact shape specified by grid_size
    binary_map = binary_map[:grid_size[0], :grid_size[1]]

    binary_map[binary_map > 0] = 1  # Convert the histogram to binary map
    return binary_map

def determine_range_and_grid_size(df, grid_resolution=500):
    if df is None or df.empty:
        print('No data found in the DataFrame.')
        return (-100, 100), (-100, 100), (grid_resolution, grid_resolution)
    min_y, max_y = df['Y'].min(), df['Y'].max()
    min_z, max_z = df['Z'].min(), df['Z'].max()
    y_range = (min_y, max_y)
    z_range = (min_z, max_z)
    grid_size_y = max(1, int((max_y - min_y) / (max_y - min_y) * grid_resolution))
    grid_size_z = max(1, int((max_z - min_z) / (max_z - min_z) * grid_resolution))
    return y_range, z_range, (grid_size_y, grid_size_z)

def run_ideal_mesh_sonar_scan_simulation(mesh_paths, axis, position, sonar_positions, angles, max_range, angle_width, num_rays):
    meshes = [pv.read(path) for path in mesh_paths]
    slice_dfs = []
    for mesh in meshes:
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        mesh.points = mesh.points.dot(rotation_matrix)
        slice_df = extract_2d_slice_from_mesh(mesh, position, axis)
        if slice_df is not None:
            slice_dfs.append(slice_df)

    if slice_dfs:
        min_y, max_y = np.inf, -np.inf
        min_z, max_z = np.inf, -np.inf
        max_resolution = -np.inf

        for df in slice_dfs:
            y_range, z_range, grid_size = determine_range_and_grid_size(df)
            min_y = min(min_y, y_range[0])
            max_y = max(max_y, y_range[1])
            min_z = min(min_z, z_range[0])
            max_z = max(max_z, z_range[1])
            max_resolution = max(max_resolution, grid_size[0], grid_size[1])
            print(f"Current DF ranges: Y({y_range}), Z({z_range}) and grid size: {grid_size}")

        y_range = (min_y, max_y)
        z_range = (min_z, max_z)
        grid_size = (max_resolution, max_resolution)
        combined_binary_map = np.zeros(grid_size)

        print(f"Final grid size: {grid_size} and ranges Y: {y_range}, Z: {z_range}")

        for df in slice_dfs:
            binary_map = create_binary_map(df, grid_size, y_range, z_range)
            if binary_map is not None:
                if binary_map.shape == combined_binary_map.shape:
                    combined_binary_map += binary_map
                else:
                    print(f"Warning: Mismatch in binary map dimensions for DF with grid size {binary_map.shape} vs combined {combined_binary_map.shape}")

        combined_binary_map[combined_binary_map > 0] = 1  # Ensure the combined map is binary

        # Visualize the binary map
        plt.figure(figsize=(10, 10))
        plt.imshow(combined_binary_map.T, extent=(y_range[0], y_range[1], z_range[0], z_range[1]), origin='lower', cmap='gray')
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.title(f'Binary Map of 2D Slices')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
