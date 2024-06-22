import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load and transform the mesh
terrain = pv.read('/Users/eirikvarnes/code/totalenergies/simulation_test/blender_terrain_test_1.obj')
rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
terrain.points = terrain.points.dot(rotation_matrix)

def extract_2d_slice_from_mesh(mesh, axis='z', position=0):
    # Define axis normals
    axes = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
    normal = axes[axis]

    # Perform the slicing
    slice = mesh.slice(normal=normal, origin=(0, 0, position))

    if slice.n_points == 0:
        print(f"No points found in the slice at {axis}={position}")
        return None

    # Extract the slice points
    points = slice.points
    df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])

    return df

# User inputs
axis = 'x'  # Choose from 'x', 'y', 'z'
position = 5  # Position along the chosen axis

# Extract the data
slice_df = extract_2d_slice_from_mesh(terrain, axis, position)
if slice_df is not None:
    # Plot using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(slice_df['X'], slice_df['Y'], slice_df['Z'], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(f'2D Slice of the Mesh at {axis}={position}')
    plt.show()
else:
    print("No slice data available to display.")
