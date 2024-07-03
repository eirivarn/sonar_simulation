import pyvista as pv
import numpy as np
import pandas as pd
from ideal_simulation.multiple_sonar import ray_cast, plot_both_views
import matplotlib.pyplot as plt

def extract_2d_slice_from_mesh(mesh, position, axis='x'):
    if axis not in ['x', 'y', 'z']:
        raise ValueError(f"Invalid axis '{axis}', must be 'x', 'y', or 'z'")
    axes = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
    origins = {'x': (position, 0, 0), 'y': (0, position, 0), 'z': (0, 0, position)}
    normal = axes[axis]
    origin = origins[axis]
    slice = mesh.slice(normal=normal, origin=origin)
    if slice.n_points == 0:
        print(f"No points found in the slice at {axis}={position}")
        return None
    points = slice.points*100
    df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
    return df

def assign_mesh_id(df, mesh_id):
    df['Mesh_ID'] = mesh_id  # Assign a unique identifier to each mesh's points
    return df

def create_label_map(df, grid_size, x_range, y_range):
    if df is None:
        return None

    margin_x = (x_range[1] - x_range[0]) * 0.01
    margin_y = (y_range[1] - y_range[0]) * 0.01

    x_bins = np.linspace(x_range[0] - margin_x, x_range[1] + margin_x, grid_size[0] + 1)
    y_bins = np.linspace(y_range[0] - margin_y, y_range[1] + margin_y, grid_size[1] + 1)

    label_map = np.zeros(grid_size)

    for _, row in df.iterrows():
        y_bin = np.digitize(row['Y'], x_bins) - 1
        z_bin = np.digitize(row['Z'], y_bins) - 1
        if 0 <= y_bin < grid_size[0] and 0 <= z_bin < grid_size[1]:
            label_map[y_bin, z_bin] = row['Mesh_ID']

    return label_map

def determine_range_and_grid_size(df, grid_resolution=(200, 200)):
    if df is None or df.empty:
        return (-100, 100), (-100, 100), (grid_resolution, grid_resolution)

    min_y, max_y = df['Y'].min(), df['Y'].max()
    min_z, max_z = df['Z'].min(), df['Z'].max()
    
    y_range = (min_y, max_y)
    z_range = (min_z, max_z)

    return y_range, z_range, grid_resolution

def plot_label_map(label_map, y_range, z_range, title='Label Map of 2D Slices'):
    plt.figure(figsize=(12, 4))
    unique_labels = np.unique(label_map)
    num_colors = int(unique_labels.max() - unique_labels.min() + 1)
    cmap = plt.get_cmap('viridis', num_colors)
    plt.imshow(label_map.T, extent=(y_range[0], y_range[1], z_range[0], z_range[1]), origin='lower', cmap=cmap)
    plt.colorbar(ticks=np.arange(unique_labels.min(), unique_labels.max() + 1))
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def run_ideal_mesh_sonar_scan_simulation(mesh_paths, axis, position, sonar_positions, angles, max_range, angle_width, num_rays):
    if not all(isinstance(path, str) for path in mesh_paths):
        raise ValueError("All mesh paths must be strings.")
    if axis not in ['x', 'y', 'z']:
        raise ValueError(f"Invalid axis '{axis}', must be 'x', 'y', or 'z'")

    meshes = []
    for path in mesh_paths:
        try:
            mesh = pv.read(path)
            meshes.append(mesh)
        except Exception as e:
            print(f"Error reading mesh from {path}: {e}")
            continue

    slice_dfs = []
    for mesh_index, mesh in enumerate(meshes):
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        mesh.points = mesh.points.dot(rotation_matrix)
        slice_df = extract_2d_slice_from_mesh(mesh, position, axis)
        if slice_df is not None:
            slice_df = assign_mesh_id(slice_df, mesh_index + 1)
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

        # Add padding to the z_range to accommodate sonars above
        padding_factor = 3  # Factor to extend the z_range
        padded_z_max = max_z + (max_z - min_z) * padding_factor
        padded_z_range = (min_z, padded_z_max)

        # Create a larger grid to accommodate the padded z_range
        padded_grid_size = (grid_size[0], int(grid_size[1] * (1 + padding_factor)))
        combined_label_map = np.zeros(padded_grid_size)

        print(f"Final grid size: {padded_grid_size} and ranges Y: {y_range}, Z: {padded_z_range}")

        # Insert the existing label map into the appropriate portion of the larger grid
        for df in slice_dfs:        
            label_map = create_label_map(df, grid_size, y_range, z_range)
            if label_map is not None:
                combined_label_map[:, :grid_size[1]] = np.maximum(combined_label_map[:, :grid_size[1]], label_map)

        plot_label_map(combined_label_map, y_range, padded_z_range, title='Combined Label Map of All Meshes')
        print(f"Combined label map created with shape: {combined_label_map.shape} and ranges Y: {y_range}, Z: {padded_z_range}")

        all_sonar_data, all_theta = [], []
        for pos, angle in zip(sonar_positions, angles):
            sonar_data, theta = ray_cast(combined_label_map.T, pos, angle, max_range, angle_width, num_rays, y_range, padded_z_range)
            all_sonar_data.append(sonar_data)
            all_theta.append(theta)

        transformed_coords = plot_both_views(combined_label_map.T, y_range, padded_z_range, sonar_positions, all_sonar_data, angles, angle_width, max_range, all_theta, True)
        return transformed_coords
    else:
        print("No slice data available for display.")