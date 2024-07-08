from typing import Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from ideal_simulation.pipeline_seafloor_analysis import *

def draw_circle(ax, x_circle, y_circle, z, radius, stability_percentage):
    # Generate points for the circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta) + x_circle
    y = radius * np.sin(theta) + z
    z_circle = np.full_like(x, y_circle)
    ax.plot(x, z_circle, y, color=plt.cm.brg(stability_percentage / 100), alpha=0.3)

def run_3d_seafloor_analysis(sonar_positions: List[Tuple[int, int]], angles: List[int], slice_positions: List[int]) -> List[Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, float, float, float]]]:
    results = []
    for slice_position in slice_positions:
        result = run_pipeline_seafloor_detection(slice_position, sonar_positions, angles, get_ground_truth=True)
        results.append(result)
    return results

def plot_pipe(results: List[Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, float, float, float]]], slice_positions: List[int]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pipe_x = []
    pipe_y = []
    pipe_z = []
    colors = []

    for i, result in enumerate(results):
        if result is not None:
            x_circle, y_circle, radius, _, _, _, stability_percentage, _ = result
            slice_position = slice_positions[i]
            pipe_x.append(x_circle)
            pipe_y.append(slice_position)
            pipe_z.append(y_circle)
            colors.append(stability_percentage)

    scatter = ax.scatter(pipe_x, pipe_y, pipe_z, c=colors, cmap='brg', label='Pipe Center', s=20)
    
    for xc, yc, z, radius, color in zip(pipe_x, pipe_y, pipe_z, [res[2] for res in results if res is not None], colors):
        # Generate points for the circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = radius * np.cos(theta) + xc
        y_circle = radius * np.sin(theta) + z
        z_circle = np.full_like(x_circle, yc)
        ax.plot(x_circle, z_circle, y_circle, color=plt.cm.brg(color/100))

    ax.set_xlabel('X Position')
    ax.set_ylabel('Slice Position')
    ax.set_zlabel('Y Height')
    fig.colorbar(scatter, ax=ax, label='Stability Percentage')
    plt.legend()
    plt.show()

def run_3d_mapping_simulation(sonar_positions: List[Tuple[int, int]], angles: List[int], slice_positions: List[int]) -> None:
    results = run_3d_seafloor_analysis(sonar_positions, angles, slice_positions)
    plot_pipe(results, slice_positions)