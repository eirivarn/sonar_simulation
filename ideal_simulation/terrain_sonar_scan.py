import pyvista as pv
import numpy as np
import pandas as pd
from ideal_simulation.multiple_sonar import plot_both_views, ray_cast
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

def plot_and_return_label_map(label_map, y_range, z_range, title='Label Map of 2D Slices', resolution=1):
    """
    Plots the label map and returns a rescaled version of the map based on the specified resolution.
    
    :param label_map: numpy.ndarray, the original label map to plot.
    :param y_range: tuple, the physical y-range (min_y, max_y) to scale to.
    :param z_range: tuple, the physical z-range (min_z, max_z) to scale to.
    :param title: str, title of the plot.
    :param resolution: float, resolution in units per pixel; determines the granularity of the returned map.
    :return: numpy.ndarray, the rescaled label map.
    """
    plt.figure(figsize=(12, 4))
    unique_labels = np.unique(label_map)
    num_colors = int(unique_labels.max() - unique_labels.min() + 1)
    cmap = plt.get_cmap('viridis', num_colors)
    
    # Define the dimensions of the new rescaled map
    y_size = int((y_range[1] - y_range[0]) / resolution)
    z_size = int((z_range[1] - z_range[0]) / resolution)
    rescaled_map = np.zeros((z_size, y_size))
    
    # Create interpolation grid
    y_original = np.linspace(y_range[0], y_range[1], label_map.shape[1])
    z_original = np.linspace(z_range[0], z_range[1], label_map.shape[0])
    y_new = np.linspace(y_range[0], y_range[1], y_size)
    z_new = np.linspace(z_range[0], z_range[1], z_size)
    
    # Interpolate using griddata
    from scipy.interpolate import griddata
    points = np.meshgrid(y_original, z_original)
    values = label_map.flatten()
    grid_z, grid_y = np.meshgrid(z_new, y_new)
    rescaled_map = griddata((points[0].flatten(), points[1].flatten()), values, (grid_y, grid_z), method='nearest')
    
    # Plotting
    plt.imshow(rescaled_map, extent=(y_range[0], y_range[1], z_range[0], z_range[1]), origin='lower', cmap=cmap)
    plt.colorbar(ticks=np.arange(unique_labels.min(), unique_labels.max() + 1))
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return rescaled_map

def create_label_map(df, grid_size, x_range, y_range):
    if df is None:
        return None

    # Compute margins for binning
    margin_x = (x_range[1] - x_range[0]) * 0.01
    margin_y = (y_range[1] - y_range[0]) * 0.01

    # Define bin edges with margins
    x_bins = np.linspace(x_range[0] - margin_x, x_range[1] + margin_x, grid_size[0] + 1)
    y_bins = np.linspace(y_range[0] - margin_y, y_range[1] + margin_y, grid_size[1] + 1)

    label_map = np.zeros(grid_size)

    for _, row in df.iterrows():
        x_bin = np.digitize(row['Y'] - x_range[0], x_bins) - 1
        y_bin = np.digitize(row['Z'] - y_range[0], y_bins) - 1
        if 0 <= x_bin < grid_size[0] and 0 <= y_bin < grid_size[1]:
            label_map[x_bin, y_bin] = row['Mesh_ID']

    return label_map

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
    min_y, max_y, min_z, max_z = np.inf, -np.inf, np.inf, -np.inf  # Initialize to find global extents

    # Process each mesh to calculate global extents
    for mesh_index, mesh in enumerate(meshes):
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        mesh.points = mesh.points.dot(rotation_matrix)
        slice_df = extract_2d_slice_from_mesh(mesh, position, axis)
        if slice_df is not None:
            slice_df = assign_mesh_id(slice_df, mesh_index + 1)
            min_y, max_y = min(min_y, slice_df['Y'].min()), max(max_y, slice_df['Y'].max())
            min_z, max_z = min(min_z, slice_df['Z'].min()), max(max_z, slice_df['Z'].max())
            slice_dfs.append(slice_df)

    # Normalize the Y coordinates to start from zero
    y_range = (0, max_y - min_y)
    
    # Calculate padding for Z range and normalize
    padding = (max_z - min_z) * 3
    z_range = (0, max_z - min_z)  # Normalized Z range without padding
    padded_z_range = (0, z_range[1] + padding)  # Z range with padding

    grid_size = (400, 400)  

    combined_label_map = np.zeros(grid_size)

    # Normalized slice placement and combined label map creation code...
    for df in slice_dfs:
        # Normalize coordinates in the DataFrame
        df['Y'] -= min_y
        df['Z'] -= min_z

        label_map = create_label_map(df, grid_size, y_range, padded_z_range)
        if label_map is not None:
            combined_label_map = np.maximum(combined_label_map, label_map)
    


    label_map = plot_and_return_label_map(combined_label_map, y_range, padded_z_range, title='Combined Label Map of All Meshes')
    print(f"Combined label map created with shape: {label_map.shape} and ranges Y: {y_range}, Z: {z_range}")

    all_sonar_data, all_theta = [], []
    for pos, angle in zip(sonar_positions, angles):
        sonar_data, theta = ray_cast(label_map, pos, angle, max_range, angle_width, num_rays, y_range, padded_z_range)
        all_sonar_data.append(sonar_data)
        all_theta.append(theta)

    transformed_coords = plot_both_views(label_map, y_range, padded_z_range, sonar_positions, all_sonar_data, angles, angle_width, max_range, all_theta, True)
    return transformed_coords
