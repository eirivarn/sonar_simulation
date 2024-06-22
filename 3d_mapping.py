import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import pandas as pd

# Load and transform the mesh
terrain = pv.read('/Users/eirikvarnes/code/totalenergies/simulation_test/blender_terrain_test_1.obj')
rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
terrain.points = terrain.points.dot(rotation_matrix)

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

def plot_both_views(room, pos, sonar_data, angle, angle_width, max_range, theta):
    """ Plot both room view and sonar image view as a cone in polar coordinates. """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for traditional room view
    ax1.imshow(room, cmap='gray', origin='lower', interpolation='bilinear')
    ax1.scatter([pos[1]], [pos[0]], color='red')  # Sonar position
    num_rays = len(sonar_data)
    for (r, strength), t in zip(sonar_data, theta):
        x = pos[0] + r * np.cos(t)
        y = pos[1] + r * np.sin(t)
        ax1.plot([pos[1], y], [pos[0], x], 'yellow')
    ax1.set_title('2D Slice with Sonar Simulation')

    # Calculate relative angles to sonar
    relative_theta = [t - np.radians(angle) for t in theta]

    # Plot for sonar image view as a cone
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_theta_zero_location('S')  # Set zero angle to the top (straight up)
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, max_range)
    ax2.set_xlim(-np.radians(angle_width / 2), np.radians(angle_width / 2))  # Center the sonar field of view
    ax2.set_title('Sonar Image')
    ax2.set_facecolor('white')

    colors = viridis(np.linspace(0, 1, max_range))
    for (r, strength), t in zip(sonar_data, relative_theta):
        color = colors[int(r * strength)]
        ax2.scatter(t, r, color=color, s=10 * strength + 1)

    plt.show()


positions = np.arange(-26, 26.1, 0.1)
all_sonar_hits = []

for position in positions:
    slice_df = extract_2d_slice_from_mesh(terrain, position, axis='x')
    if slice_df is not None:
        binary_map = create_binary_map_from_slice((1000, 1000), slice_df)
        pos = (500, 500)  # Sonar position on the map
        sonar_hits = ray_cast(binary_map, pos, 180, 1000, 60, 100)
        for hit in sonar_hits:
            all_sonar_hits.append((position, *hit))  # Save with position

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