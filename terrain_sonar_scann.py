import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import pandas as pd
from basic_sonar import ray_cast, plot_both_views

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


def create_binary_map_from_slice(dimensions, slice_df):
    """ Create a binary map from slice data. """
    binary_map = np.zeros(dimensions)
    min_x, max_x = slice_df['Z'].min(), slice_df['Z'].max()  # Use Z instead of X
    min_y, max_y = slice_df['Y'].min(), slice_df['Y'].max()

    print(f"min_y: {min_x}, max_x (Z): {max_x}")
    print(f"min_x: {min_y}, max_y: {max_y}")

    # Calculate scaling factors
    scale_x = (dimensions[0] / (max_x - min_x))/6
    scale_y = (dimensions[1] / (max_y - min_y))

    for _, row in slice_df.iterrows():
        x, y = row['Z'], row['Y']  # Use Z instead of X
        x_index = int((x - min_x) * scale_x+800)
        y_index = int((y - min_y) * scale_y)
        x_index = dimensions[0] - 1 - x_index  # Flip x-coordinate
        y_index = dimensions[1] - 1 - y_index  # Flip y-coordinate
        
        if 0 <= x_index < dimensions[0] and 0 <= y_index < dimensions[1]:
            binary_map[x_index, y_index] = 1
        else:
            print(f"Out of bounds: x_index={x_index}, y_index={y_index}")

    return binary_map


# For testing
"""""
# Load and transform the mesh
terrain = pv.read('/Users/eirikvarnes/code/totalenergies/simulation_test/blender_terrain_test_1.obj')
rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
terrain.points = terrain.points.dot(rotation_matrix)

# Define room dimensions
dimensions = (1000, 1000)  # Increase dimensions to capture wider terrain

# User inputs for slice extraction
axis = 'x'  # Choose from 'x', 'y', 'z'
position = -25  # Position along the chosen axis

# Extract the data
slice_df = extract_2d_slice_from_mesh(terrain, position, axis)

if slice_df is not None:
    # Create binary map with the slice data (using Z for X)
    binary_map = create_binary_map_from_slice(dimensions, slice_df)
    
    # Define sonar parameters
    pos = (500, 500)  # Update position to be outside the room
    angle = 180  # direction in degrees (mid-point direction pointing right)
    max_range = 1000  # Increase max range to fit new dimensions
    angle_width = 60  # total sonar angle width in degrees
    num_rays = 100  # number of rays for higher resolution

    # Perform ray-casting
    sonar_data, theta = ray_cast(binary_map, pos, angle, max_range, angle_width, num_rays)

    # Visualize both views
    plot_both_views(binary_map, pos, sonar_data, angle, angle_width, max_range, theta)
else:
    print("No slice data available to display.")
"""""