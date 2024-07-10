from typing import Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from ideal_simulation.pipeline_seafloor_analysis import *
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
        result = run_pipeline_seafloor_detection(slice_position, sonar_positions, angles, get_ground_truth=False)
        results.append(result)
    return results

def plot_pipe(results: List[Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, float, float, float]]], slice_positions: List[int]) -> None:
    fig = plt.figure(figsize=(12, 8))  # Adjust the figsize to make the figure larger
    ax = fig.add_subplot(111, projection='3d')

    pipe_x = []
    pipe_y = []
    pipe_z = []
    colors = []

    scale_factor = 5

    for i, result in enumerate(results):
        if result is not None:
            # Check length and unpack accordingly
            if len(result) == 10:
                x_circle, y_circle, radius, curve_x, curve_y, free_span_status, stability_percentage, enclosed_percentage, relative_distance, angle_degrees = result
            elif len(result) == 11:
                x_circle, y_circle, radius, curve_x, curve_y, free_span_status, stability_percentage, enclosed_percentage, relative_distance, angle_degrees, additional_data = result

            slice_position = slice_positions[i]
            pipe_x.append(x_circle)
            pipe_y.append(slice_position)
            pipe_z.append(y_circle)
            colors.append(stability_percentage)

            # Plot the curve with a colormap based on height
            curve_z = np.full_like(curve_x, slice_position * scale_factor)  # Scale the slice positions by the factor
            norm = plt.Normalize(vmin=np.min(curve_y), vmax=np.max(curve_y))
            points = np.array([curve_x, curve_z, curve_y]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(segments, cmap='viridis', norm=norm, alpha=0.5)
            lc.set_array(curve_y)
            ax.add_collection(lc)

    scatter = ax.scatter(pipe_x, np.array(pipe_y) * scale_factor, pipe_z, c=colors, cmap='brg', label='Pipe Center', s=20)

    for xc, yc, z, radius, color in zip(pipe_x, pipe_y, pipe_z, [res[2] for res in results if res is not None], colors):
        # Generate points for the circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = radius * np.cos(theta) + xc
        y_circle = radius * np.sin(theta) + z
        z_circle = np.full_like(x_circle, yc * scale_factor)
        ax.plot(x_circle, z_circle, y_circle, color=plt.cm.brg(color / 100), alpha=0.3)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Slice Position (scaled by 5)')
    ax.set_zlabel('Y Height')
    fig.colorbar(scatter, ax=ax, label='Stability Percentage')
    plt.legend()

    # Adjust the aspect ratio
    ax.set_box_aspect([1, scale_factor, 1])

    # Adjust subplot size
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust these values to control the subplot size

    plt.show()

def run_3d_mapping_simulation(sonar_positions: List[Tuple[int, int]], angles: List[int], slice_positions: List[int]) -> None:
    results = run_3d_seafloor_analysis(sonar_positions, angles, slice_positions)
    plot_pipe(results, slice_positions)
    return results