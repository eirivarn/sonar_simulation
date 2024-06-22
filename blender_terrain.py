import pyvista as pv
import numpy as np

# Load OBJ file
terrain = pv.read('/Users/eirikvarnes/code/totalenergies/simulation_test/blender_terrain_test_1.obj')

# Define rotation matrix to rotate Y-axis to Z-axis
# This rotates by -90 degrees around the X-axis
rotation_matrix = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])

# Apply rotation to vertex coordinates
points = terrain.points.dot(rotation_matrix)

# Update the mesh with the rotated coordinates
terrain.points = points

# Visualize the transformed terrain
p = pv.Plotter()
p.add_mesh(terrain, color='tan')
p.show()

point_cloud = pv.PolyData(points)

# Visualize the point cloud
p = pv.Plotter()
p.add_points(point_cloud, color='tan', point_size=5)  
p.show()