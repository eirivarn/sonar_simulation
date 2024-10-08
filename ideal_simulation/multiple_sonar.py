import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from typing import List, Tuple, Union
from ideal_simulation.basic_sonar import create_room_with_pipe_and_ground, ground_wave_function, ray_cast
from config import config
from concurrent.futures import ThreadPoolExecutor

def transform_to_global(pos: Tuple[int, int], sonar_data: List[Tuple[int, int]], theta: List[float]) -> List[Tuple[float, float, int]]:
    theta = np.array(theta)
    r, strength = zip(*sonar_data)
    r = np.array(r)
    x = pos[0] + r * np.cos(theta)
    y = pos[1] + r * np.sin(theta)
    return list(zip(x, y, strength))

def transform_to_reference_sonar(ref_pos: Tuple[int, int], ref_angle: float, global_coords: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    """Transform global coordinates to the reference sonar's coordinate system using vectorized operations."""
    # Convert list of tuples to a structured NumPy array for vectorized operations
    global_coords = np.array(global_coords, dtype=[('x', float), ('y', float), ('strength', int)])
    
    # Calculate rotation matrix components
    ref_angle_rad = np.radians(ref_angle)
    cos_angle = np.cos(-ref_angle_rad)
    sin_angle = np.sin(-ref_angle_rad)
    
    # Subtract reference position (broadcasting)
    dx = global_coords['x'] - ref_pos[0]
    dy = global_coords['y'] - ref_pos[1]
    
    # Apply the rotation matrix
    transformed_x = dx * cos_angle - dy * sin_angle
    transformed_y = dx * sin_angle + dy * cos_angle
    
    # Convert to polar coordinates
    transformed_r = np.sqrt(transformed_x**2 + transformed_y**2)
    transformed_theta = np.arctan2(transformed_y, transformed_x)
    
    return list(zip(transformed_r, transformed_theta, global_coords['strength']))

def find_farthest_points(sonar_positions: List[Tuple[int, int]]) -> Tuple[Union[Tuple[int, int], None], Union[Tuple[int, int], None]]:
    max_distance: float = 0
    point1: Union[Tuple[int, int], None] = None
    point2: Union[Tuple[int, int], None] = None

    if len(sonar_positions) < 2:
        if len(sonar_positions) == 1:
            return sonar_positions[0], sonar_positions[0]
        return None, None

    for i in range(len(sonar_positions)):
        for j in range(i + 1, len(sonar_positions)):
            distance: float = np.linalg.norm(np.array(sonar_positions[i]) - np.array(sonar_positions[j]))
            if distance > max_distance:
                max_distance = distance
                point1 = sonar_positions[i]
                point2 = sonar_positions[j]
    return point1, point2

def plot_both_views(world: np.ndarray, y_range: Tuple[int, int], z_range: Tuple[int, int], sonar_positions: List[Tuple[int, int]], all_sonar_data: List[List[Tuple[int, int]]], all_theta: List[List[float]]) -> List[Tuple[float, float, int]]:
    plot_size: List[int] = config.get('plotting', 'plot_size')
    room_color: str = config.get('plotting', 'room_color')
    max_range: int = config.get('sonar', 'max_range')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_size)
    ax1.imshow(world, extent=(y_range[0], y_range[1], z_range[0], z_range[1]), cmap=room_color, origin='lower', interpolation='bilinear')
    colors: List[str] = ['red', 'blue', 'green', 'yellow']

    for idx, (pos, sonar_data, theta) in enumerate(zip(sonar_positions, all_sonar_data, all_theta)):
        ax1.scatter([pos[1]], [pos[0]], color=colors[idx % len(colors)])
        for (r, strength), t in zip(sonar_data, theta):
            y: float = pos[0] + r * np.cos(t)
            z: float = pos[1] + r * np.sin(t)
            ax1.plot([pos[1], z], [pos[0], y], color=colors[idx % len(colors)])

    ax1.set_title('Room with Pipe and Ground')
    ax1.set_xlim(y_range)
    ax1.set_ylim(z_range)

    global_coords: List[Tuple[float, float, int]] = []
    for pos, sonar_data, theta in zip(sonar_positions, all_sonar_data, all_theta):
        global_coords.extend(transform_to_global(pos, sonar_data, theta))

    point1, point2 = find_farthest_points(sonar_positions)
    if point1 and point2:
        ref_pos: List[float] = [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]
    else:
        ref_pos = [0, 0]  # Default reference position

    min_angle: float = np.min([np.min(t) for t in all_theta]) if all_theta else 0
    max_angle: float = np.max([np.max(t) for t in all_theta]) if all_theta else 0
    ref_angle: float = 0  # Default reference angle

    transformed_coords: List[Tuple[float, float, int]] = transform_to_reference_sonar(ref_pos, ref_angle, global_coords)
    
    if config.show_plots:
        ax2 = plt.subplot(122, projection='polar')
        ax2.set_theta_zero_location('S')
        ax2.set_theta_direction(-1)
        ax2.set_ylim(0, max_range)
        ax2.set_xlim(min_angle, max_angle)
        ax2.set_title('Sonar Image')
        ax2.set_facecolor('white')

        strengths = np.array([coord[2] for coord in transformed_coords])
        colors = viridis((strengths - np.min(strengths)) / (np.max(strengths) - np.min(strengths) + 1))

        for (r, t, strength), color in zip(transformed_coords, colors):
            ax2.scatter(t, r, color=color, s=50 * strength)

        plt.show()

    return [(r, t, strength) for r, t, strength in transformed_coords]

def run_ideal_multiple_sonar_simulation(sonar_positions: List[Tuple[int, int]], angles: List[float]) -> None:
    dimensions = config.dimensions
    pipe_center = config.pipe_center
    pipe_radius = config.pipe_radius
    max_range = config.get('sonar', 'max_range')
    angle_width = config.get('sonar', 'angle_width')
    num_rays = config.get('sonar', 'num_rays')

    room = create_room_with_pipe_and_ground(dimensions, pipe_center, pipe_radius, ground_wave_function)
    y_range = (0, dimensions[1])
    z_range = (0, dimensions[0])

    # Prepare to collect all sonar data and theta values
    all_sonar_data = []
    all_theta = []

    # Use ThreadPoolExecutor to parallelize sonar data collection
    with ThreadPoolExecutor(max_workers=len(sonar_positions)) as executor:
        futures = []
        for pos, angle in zip(sonar_positions, angles):
            futures.append(executor.submit(ray_cast, room, pos, angle, max_range, angle_width, num_rays))
        
        for future in futures:
            sonar_data, theta = future.result()
            all_sonar_data.append(sonar_data)
            all_theta.append(theta)
 
    plot_both_views(room, y_range, z_range, sonar_positions, all_sonar_data, all_theta)