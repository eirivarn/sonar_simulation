import pyvista as pv
import numpy as np
import pandas as pd
from ideal_simulation.multiple_sonar import ray_cast, plot_both_views


def extract_2d_slice_from_mesh(mesh, position, axis='x'):
    # Define axis normals and adjust origin dynamically
    axes = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
    origins = {'x': (position, 0, 0), 'y': (0, position, 0), 'z': (0, 0, position)}
    normal = axes[axis]
    origin = origins[axis]

    # Perform the slicing
    slice = mesh.slice(normal=normal, origin=origin)

    if slice.n_points == 0:
        print(f"No points found in the slice at {axis}={position}")
        return None

    # Extract the slice points and create DataFrame
    points = slice.points
    df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
    return df

def create_binary_map(df, grid_size, x_range, y_range):
    if df is None:
        return None

    x_bins = np.linspace(x_range[0], x_range[1], grid_size[0])
    y_bins = np.linspace(y_range[0], y_range[1], grid_size[1])

    binary_map, _, _ = np.histogram2d(df['Y'], df['Z'], bins=[x_bins, y_bins])

    # Convert the histogram to binary map (0 or 1)
    binary_map[binary_map > 0] = 1

    return binary_map


def determine_range_and_grid_size(df, grid_resolution=500):
    if df is None or df.empty:
        return (-100, 100), (-100, 100), (grid_resolution, grid_resolution) 

    min_y, max_y = df['Z'].min(), df['Z'].max()
    min_x, max_x = df['Y'].min(), df['Y'].max()
    
    y_range = (min_y, max_y)
    x_range = (min_x, max_x)

    grid_size_y = int((max_y - min_y) / (max_y - min_y) * grid_resolution)
    grid_size_x = int((max_x - min_x) / (max_x - min_x) * grid_resolution)
    
    return y_range, x_range, (grid_size_y, grid_size_x)

def run_ideal_mesh_sonar_scan_simulation(mesh_paths, axis, position, sonar_positions, angles, max_range, angle_width, num_rays):
    # Load and transform the mesh
    meshes = [pv.read(path) for path in mesh_paths]
    slice_dfs = []

    for mesh in meshes:
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        mesh.points = mesh.points.dot(rotation_matrix)
        slice_df = extract_2d_slice_from_mesh(mesh, position, axis)
        if slice_df is not None:
            slice_dfs.append(slice_df)
    
    if not slice_dfs:
        print("No slice data available to display.")
        return None

    # Initialize variables to hold the combined ranges and grid size
    combined_y_range = [float('inf'), float('-inf')]
    combined_x_range = [float('inf'), float('-inf')]
    max_resolution = 0
    
    # Process each slice to determine overall ranges and grid sizes
    for df in slice_dfs:
        y_range, x_range, grid_size = determine_range_and_grid_size(df)
        combined_y_range = [min(combined_y_range[0], y_range[0]), max(combined_y_range[1], y_range[1])]
        combined_x_range = [min(combined_x_range[0], x_range[0]), max(combined_x_range[1], x_range[1])]
        max_resolution = max(max_resolution, grid_size[1], grid_size[0])
    
    # Set a uniform grid size based on the maximum resolution found
    uniform_grid_size = (max_resolution, max_resolution)

    # Create binary maps for each slice and combine them
    combined_binary_map = np.zeros(uniform_grid_size)
    for df in slice_dfs:
        binary_map = create_binary_map(df, uniform_grid_size, combined_x_range, combined_y_range)
        if binary_map is not None:
            combined_binary_map[:binary_map.shape[1], :binary_map.shape[0]] += binary_map
    
    # Ensure the combined map is binary
    combined_binary_map[combined_binary_map > 0] = 1

    # Store all sonar data and angles for visualization
    all_sonar_data = []
    all_theta = []

    # Perform ray-casting for each sonar
    for pos, angle in zip(sonar_positions, angles):
        sonar_data, theta = ray_cast(combined_binary_map.T, pos, angle, max_range, angle_width, num_rays)
        all_sonar_data.append(sonar_data)
        all_theta.append(theta)

    x_values = []
    y_values = []
    
    # Visualize both views
    transformed_coords = plot_both_views(combined_binary_map.T, sonar_positions, all_sonar_data, angles, angle_width, max_range, all_theta, True)
    
    for (r, t, strength) in transformed_coords:
        if -np.radians(angle_width / 2) <= t <= np.radians(angle_width / 2):
            x = np.array(r * np.sin(t))
            x_values.append(x)
            y = np.array(r * np.cos(t))
            y_values.append(y)
            
    return x_values, y_values