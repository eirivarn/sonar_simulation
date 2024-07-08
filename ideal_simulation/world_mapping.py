import numpy as np
import os
from config import config
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from ideal_simulation.pipeline_seafloor_analysis import run_pipeline_seafloor_detection


def run_3d_seafloor_analysis(sonar_positions, angles):
    slice_positions = config.get('mesh_processing', 'slice_positions')
    results = []
    for slice_position in slice_positions:
        result = run_pipeline_seafloor_detection(slice_position, sonar_positions, angles)
        if result:
            x_circle, y_circle, radius, curve_x, curve_y, free_span_status, stability_percentage = result
            results.append({
                'x': x_circle, 'y': y_circle, 'slice_position': slice_position, 'radius': radius,
                'free_span_status': free_span_status, 'stability_percentage': stability_percentage
            })
        else:
            print(f"No data for slice at depth {slice_position}")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for result in results:
        # Color circle based on free-span status
        color = 'green' if result['free_span_status'] == 'Free-span' else 'red'
        circle = plt.Circle((result['x'], result['y']), result['radius'], color=color, alpha=0.5)
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=result['z'], zdir="z")
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth')
    plt.show()
    
    return results
