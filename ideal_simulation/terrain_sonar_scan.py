import numpy as np
import pyvista as pv
import pandas as pd
from typing import List, Tuple
from config import config
from ideal_simulation.multiple_sonar import plot_both_views, ray_cast
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from concurrent.futures import ThreadPoolExecutor

def process_mesh(path: str, position: float, axis: str, mesh_index: int, rotation_matrix: np.ndarray, min_y: float, min_z: float) -> Tuple[pd.DataFrame, float, float, float, float]:
    try:
        # Determine the file type from the extension
        _, file_extension = os.path.splitext(path)
        
        if file_extension.lower() == '.obj':
            # Handling OBJ files with PyVista
            mesh = pv.read(path)
        elif file_extension.lower() == '.csv':
            # Handling CSV files
            data = pd.read_csv(path)
            points = data[['X', 'Y', 'Z']].to_numpy()
            # Optionally create a mesh if needed using Delaunay triangulation
            cloud = pv.PolyData(points)
            mesh = cloud.delaunay_3d()
            mesh = mesh.extract_geometry()  # Extract surface mesh from delaunay tetrahedralization
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Apply rotation matrix to the points of the mesh
        mesh.points = mesh.points.dot(rotation_matrix)

        # Extract a 2D slice from the mesh
        slice_df = extract_2d_slice_from_mesh(mesh, position, axis)
        if slice_df is not None:
            slice_df = assign_mesh_id(slice_df, mesh_index + 1)
            return slice_df, slice_df['Y'].min(), slice_df['Y'].max(), slice_df['Z'].min(), slice_df['Z'].max()
    except Exception as e:
        print(f"Error processing mesh from {path}: {e}")
    return None, min_y, min_y, min_z, min_z

def extract_2d_slice_from_mesh(mesh: pv.PolyData, position: float, axis: str) -> pd.DataFrame:
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

def assign_mesh_id(df: pd.DataFrame, mesh_id: int) -> pd.DataFrame:
    df['Mesh_ID'] = mesh_id
    return df

def create_label_map(df: pd.DataFrame, grid_size: Tuple[int, int], x_range: Tuple[float, float], y_range: Tuple[float, float]) -> np.ndarray:
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

def plot_and_return_label_map(label_map: np.ndarray, y_range: Tuple[int, int], z_range: Tuple[int, int], title: str = 'Label Map of 2D Slices', resolution: int = 1) -> np.ndarray:
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

    if config.show_plots:
        plt.figure(figsize=(12, 6))
        plt.imshow(rescaled_map.T, cmap='viridis', origin='lower')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Y Dimension')
        plt.ylabel('Z Dimension')
        plt.show()

    return rescaled_map.T

def transform_and_plot_coordinates(transformed_coords: List[Tuple[float, float, int]], y_range: Tuple[int, int], z_range: Tuple[int, int]) -> List[Tuple[float, float, int]]:
    cartesian_coords = []
    
    for (r, theta, strength) in transformed_coords:
        if strength == 0:
            continue
        y = r * np.cos(theta)
        x = r * np.sin(theta)
        cartesian_coords.append((x, y, strength))
    
    if config.show_plots:
        # Plotting the Cartesian coordinates
        x_coords = [coord[0] for coord in cartesian_coords]
        y_coords = [coord[1] for coord in cartesian_coords]
        strengths = [coord[2] for coord in cartesian_coords]

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_coords, y_coords, c=strengths, cmap='viridis', s=10 * np.array(strengths), alpha=0.75)
        plt.colorbar(scatter, label='Strength')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Cartesian Coordinates with Strengths')

        # Calculate the aspect ratio from the y and z ranges
        y_range_span = y_range[1] - y_range[0]
        z_range_span = z_range[1] - z_range[0]
        aspect_ratio = y_range_span / z_range_span
        
        plt.gca().set_aspect(aspect_ratio, adjustable='box')
        plt.show()
    
    return cartesian_coords

def run_ideal_mesh_sonar_scan_simulation(slice_position: int, sonar_positions: List[Tuple[int, int]], angles: List[float]) -> Tuple[list, np.ndarray]:
    mesh_paths = config.separate_mesh_paths
    axis = config.get('mesh_processing', 'slice_axis')
    max_range = config.get('sonar', 'max_range')
    angle_width = config.get('sonar', 'angle_width')
    num_rays = config.get('sonar', 'num_rays')

    if not all(isinstance(path, str) for path in mesh_paths):
        raise ValueError("All mesh paths must be strings.")
    if axis not in config.get('mesh_processing', 'slice_axes'):
        raise ValueError(f"Invalid axis '{axis}', must be one of {config.get('mesh_processing', 'slice_axes')}")

    rotation_matrix = np.array(config.get('mesh_processing', 'rotation_matrix'))

    min_y, max_y, min_z, max_z = np.inf, -np.inf, np.inf, -np.inf
    slice_data_frames = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_mesh, path, slice_position, axis, idx, rotation_matrix, min_y, min_z) for idx, path in enumerate(mesh_paths)]
        for future in futures:
            slice_data_frame, mesh_min_y, mesh_max_y, mesh_min_z, mesh_max_z = future.result()
            if slice_data_frame is not None:
                slice_data_frames.append(slice_data_frame)
                min_y, max_y = min(min_y, mesh_min_y), max(max_y, mesh_max_y)
                min_z, max_z = min(min_z, mesh_min_z), max(max_z, mesh_max_z)

    y_range = (0, max_y - min_y)
    padding_factor = config.get('mesh_processing', 'padding_factor')
    padding = (max_z - min_z) * padding_factor
    z_range = (0, max_z - min_z)
    padded_z_range = (0, z_range[1] + padding)

    grid_size = config.get('mesh_processing', 'grid_size')
    combined_label_map = np.zeros(grid_size)

    for data_frame in slice_data_frames:
        data_frame['Y'] -= min_y
        data_frame['Z'] -= min_z
        label_map = create_label_map(data_frame, grid_size, y_range, padded_z_range)
        if label_map is not None:
            combined_label_map = np.maximum(combined_label_map, label_map)

    label_map = plot_and_return_label_map(combined_label_map, y_range, padded_z_range, title='Combined Label Map of All Meshes')
    print(f"Combined label map created with shape: {label_map.shape} and ranges Y: {y_range}, Z: {z_range}")

    all_sonar_data, all_theta = [], []
    for pos, angle in zip(sonar_positions, angles):
        sonar_data, theta = ray_cast(label_map, pos, angle, max_range, angle_width, num_rays)
        all_sonar_data.append(sonar_data)
        all_theta.append(theta)

    transformed_coords = plot_both_views(label_map, y_range, padded_z_range, sonar_positions, all_sonar_data, all_theta)

    cartesian_coords = transform_and_plot_coordinates(transformed_coords, y_range, padded_z_range)

    return cartesian_coords, label_map
