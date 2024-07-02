from ideal_simulation.terrain_sonar_scan import *
from ideal_simulation.multiple_sonar import plot_both_views
import matplotlib.pyplot as plt
import numpy as np
import os
import pyvista as pv

def transform_to_cartesian(r, theta):
    """ Transform polar coordinates to cartesian coordinates. """
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    return x, y

def get_sonar_2d_plot(mesh_path, position, dimensions, sonar_positions, angles, max_range, angle_width, num_rays):
    """ Return a 2D plot of the sonar data with transformed theta values to match distance scale. """
    terrain = pv.read(mesh_path)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    terrain.points = terrain.points.dot(rotation_matrix)
    
    # Extract 2D slice from the mesh
    slice_df = extract_2d_slice_from_mesh(terrain, position, axis='x')
    
    if slice_df is not None:
       
        
        return theta, distances
    else:
        print("No slice data available to display.")
        return None, None