import pandas as pd
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

def extract_2d_slice_from_mesh(mesh, position, axis='y'):
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

    binary_map, _, _ = np.histogram2d(df['Z'], df['X'], bins=[x_bins, y_bins])

    # Convert the histogram to binary map (0 or 1)
    binary_map[binary_map > 0] = 1

    return binary_map

# Read the meshes
pipeline_mesh = pv.read('/Users/eirikvarnes/code/blender/pipeline.obj')
seafloor_mesh = pv.read('/Users/eirikvarnes/code/blender/seafloor.obj')

# Apply rotation
rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
pipeline_mesh.points = pipeline_mesh.points.dot(rotation_matrix)
seafloor_mesh.points = seafloor_mesh.points.dot(rotation_matrix)

# Define the position and axis for slicing
slice_position = 3.0  # Adjust as needed
slice_axis = 'x'  # Adjust as needed

# Extract slices
pipeline_slice_df = extract_2d_slice_from_mesh(pipeline_mesh, slice_position, slice_axis)
seafloor_slice_df = extract_2d_slice_from_mesh(seafloor_mesh, slice_position, slice_axis)

# Automatically determine the range and grid size
def determine_range_and_grid_size(df, grid_resolution=500):
    # Default values if DataFrame is None or empty
    if df is None or df.empty:
        return (-100, 100), (-100, 100), (grid_resolution, grid_resolution)

    # Extract the minimum and maximum values from 'Y' and 'Z' columns
    min_y, max_y = df['Y'].min(), df['Y'].max()
    min_z, max_z = df['Z'].min(), df['Z'].max()
    
    # Define the ranges for Y and Z
    y_range = (min_y, max_y)
    z_range = (min_z, max_z)

    # Calculate the grid size for each axis
    if max_y != min_y:
        grid_size_y = int(grid_resolution)
    else:
        grid_size_y = grid_resolution
    
    if max_z != min_z:
        grid_size_z = int(grid_resolution)
    else:
        grid_size_z = grid_resolution

    return y_range, z_range, (grid_size_y, grid_size_z)

# Determine range and grid size for both slices
pipeline_y_range, pipeline_z_range, pipeline_grid_size = determine_range_and_grid_size(pipeline_slice_df)
seafloor_y_range, seafloor_z_range, seafloor_grid_size = determine_range_and_grid_size(seafloor_slice_df)

# Find the combined range and grid size
y_range = (min(pipeline_y_range[0], seafloor_y_range[0]), max(pipeline_y_range[1], seafloor_y_range[1]))
z_range = (min(pipeline_z_range[0], seafloor_z_range[0]), max(pipeline_z_range[1], seafloor_z_range[1]))

# Ensure grid size is the same for both maps by taking the maximum range and resolution
max_resolution = max(pipeline_grid_size[0], pipeline_grid_size[1], seafloor_grid_size[0], seafloor_grid_size[1])
grid_size = (max_resolution, max_resolution)

# Create binary maps
pipeline_binary_map = create_binary_map(pipeline_slice_df, grid_size, y_range, z_range)
seafloor_binary_map = create_binary_map(seafloor_slice_df, grid_size, y_range, z_range)

# Combine binary maps
combined_binary_map = np.zeros(grid_size)
if pipeline_binary_map is not None:
    combined_binary_map[:pipeline_binary_map.shape[0], :pipeline_binary_map.shape[1]] += pipeline_binary_map
if seafloor_binary_map is not None:
    combined_binary_map[:seafloor_binary_map.shape[0], :seafloor_binary_map.shape[1]] += seafloor_binary_map

combined_binary_map[combined_binary_map > 0] = 1  # Ensure the combined map is binary

# Visualize the binary map
plt.figure(figsize=(10, 10))
plt.imshow(combined_binary_map.T, extent=(y_range[0], y_range[1], z_range[0], z_range[1]), origin='lower', cmap='gray')
plt.xlabel('Y')
plt.ylabel('Z')
plt.title(f'Binary Map of 2D Slices at {slice_axis}={slice_position}')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
