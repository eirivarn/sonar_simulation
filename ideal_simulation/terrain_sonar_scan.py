import pyvista as pv
import numpy as np
import pandas as pd
from ideal_simulation.multiple_sonar import plot_both_views, ray_cast
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from concurrent.futures import ThreadPoolExecutor

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
    points = slice.points * 100
    df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
    return df

def assign_mesh_id(df, mesh_id):
    df['Mesh_ID'] = mesh_id
    return df

def plot_and_return_label_map(label_map, y_range, z_range, title='Label Map of 2D Slices', resolution=1):
    y_size = int((y_range[1] - y_range[0]) / resolution)
    z_size = int((z_range[1] - z_range[0]) / resolution)
    rescaled_map = np.zeros((y_size, z_size))

    y_original = np.linspace(y_range[0], y_range[1], label_map.shape[1])
    z_original = np.linspace(z_range[0], z_range[1], label_map.shape[0])
    points = np.meshgrid(z_original, y_original)

    z_new = np.linspace(z_range[0], z_range[1], z_size)
    y_new = np.linspace(y_range[0], y_range[1], y_size)
    grid_y, grid_z = np.meshgrid(z_new, y_new)

    points_flatten = (points[0].flatten(), points[1].flatten())
    values_flatten = label_map.flatten()

    rescaled_map = griddata(points_flatten, values_flatten, (grid_y, grid_z), method='nearest')

    plt.figure(figsize=(12, 6))
    plt.imshow(rescaled_map.T, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Y Dimension')
    plt.ylabel('Z Dimension')
    plt.show()

    return rescaled_map.T

def create_label_map(df, grid_size, x_range, y_range):
    if df is None:
        return None

    margin_x = (x_range[1] - x_range[0]) * 0.01
    margin_y = (y_range[1] - y_range[0]) * 0.01

    x_bins = np.linspace(x_range[0] - margin_x, x_range[1] + margin_x, grid_size[0] + 1)
    y_bins = np.linspace(y_range[0] - margin_y, y_range[1] + margin_y, grid_size[1] + 1)

    label_map = np.zeros(grid_size, dtype=int)

    # Use vectorized operations for binning
    x_bin_indices = np.digitize(df['Y'] - x_range[0], x_bins) - 1
    y_bin_indices = np.digitize(df['Z'] - y_range[0], y_bins) - 1

    valid_indices = (0 <= x_bin_indices) & (x_bin_indices < grid_size[0]) & (0 <= y_bin_indices) & (y_bin_indices < grid_size[1])
    label_map[x_bin_indices[valid_indices], y_bin_indices[valid_indices]] = df['Mesh_ID'][valid_indices].astype(int)

    return label_map

def process_mesh(path, position, axis, mesh_index, rotation_matrix, min_y, min_z):
    try:
        mesh = pv.read(path)
        mesh.points = mesh.points.dot(rotation_matrix)
        slice_df = extract_2d_slice_from_mesh(mesh, position, axis)
        if slice_df is not None:
            slice_df = assign_mesh_id(slice_df, mesh_index + 1)
            return slice_df, slice_df['Y'].min(), slice_df['Y'].max(), slice_df['Z'].min(), slice_df['Z'].max()
    except Exception as e:
        print(f"Error reading mesh from {path}: {e}")
    return None, min_y, min_y, min_z, min_z

def run_ideal_mesh_sonar_scan_simulation(mesh_paths, axis, position, sonar_positions, angles, max_range, angle_width, num_rays):
    if not all(isinstance(path, str) for path in mesh_paths):
        raise ValueError("All mesh paths must be strings.")
    if axis not in ['x', 'y', 'z']:
        raise ValueError(f"Invalid axis '{axis}', must be 'x', 'y', or 'z'")

    rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    min_y, max_y, min_z, max_z = np.inf, -np.inf, np.inf, -np.inf
    slice_dfs = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_mesh, path, position, axis, idx, rotation_matrix, min_y, min_z) for idx, path in enumerate(mesh_paths)]
        for future in futures:
            slice_df, mesh_min_y, mesh_max_y, mesh_min_z, mesh_max_z = future.result()
            if slice_df is not None:
                slice_dfs.append(slice_df)
                min_y, max_y = min(min_y, mesh_min_y), max(max_y, mesh_max_y)
                min_z, max_z = min(min_z, mesh_min_z), max(max_z, mesh_max_z)

    y_range = (0, max_y - min_y)
    padding = (max_z - min_z) * 3
    z_range = (0, max_z - min_z)
    padded_z_range = (0, z_range[1] + padding)

    grid_size = (250, 250)
    combined_label_map = np.zeros(grid_size)

    for df in slice_dfs:
        df['Y'] -= min_y
        df['Z'] -= min_z
        label_map = create_label_map(df, grid_size, y_range, padded_z_range)
        if label_map is not None:
            combined_label_map = np.maximum(combined_label_map, label_map)

    label_map = plot_and_return_label_map(combined_label_map, y_range, padded_z_range, title='Combined Label Map of All Meshes')
    print(f"Combined label map created with shape: {label_map.shape} and ranges Y: {y_range}, Z: {z_range}")

    all_sonar_data, all_theta = [], []
    for pos, angle in zip(sonar_positions, angles):
        sonar_data, theta = ray_cast(label_map, pos, angle, max_range, angle_width, num_rays)
        all_sonar_data.append(sonar_data)
        all_theta.append(theta)

    transformed_coords = plot_both_views(label_map, y_range, padded_z_range, sonar_positions, all_sonar_data, angles, angle_width, max_range, all_theta, True)
    return transformed_coords
