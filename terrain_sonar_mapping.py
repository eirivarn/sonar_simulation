import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from terrain_sonar_scann import extract_2d_slice_from_mesh, create_binary_map_from_slice

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


#"""
# Load and transform the mesh
terrain = pv.read('/Users/eirikvarnes/code/totalenergies/simulation_test/blender_terrain_test_1.obj')
rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
terrain.points = terrain.points.dot(rotation_matrix)

positions = np.arange(-26, 26, 5)
all_sonar_hits = []

for position in positions:
    position = -position
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
#"""
