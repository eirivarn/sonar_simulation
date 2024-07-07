import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from noise import pnoise1
from typing import Tuple, Callable, List
from config import config

def create_room_with_pipe_and_ground(dimensions: Tuple[int, int], pipe_center: Tuple[int, int], pipe_radius: int, ground_wave: Callable[[int], int]) -> np.ndarray:
    room = np.zeros(dimensions)
    y, x = np.ogrid[:dimensions[0], :dimensions[1]]
    distance_from_center = np.sqrt((x - pipe_center[1])**2 + (y - pipe_center[0])**2)
    room += np.clip(1 - (distance_from_center - pipe_radius), 0, 1)
    for y in range(dimensions[1]):
        x = ground_wave(y)
        if 0 <= x < dimensions[0]:
            room[x, y] = 1
    return room

def ground_wave_function(y: int) -> int:
    """ Function to generate a wave-like ground using Perlin noise with higher resolution. """
    amplitude = config.get('ground_wave', 'amplitude')
    frequency = config.get('ground_wave', 'frequency')
    base_level = config.get('ground_wave', 'base_level')
    repeat = config.get('ground_wave', 'repeat')
    return int(base_level + amplitude * pnoise1(y * frequency, repeat=repeat))

def ray_cast(binary_map: np.ndarray, pos: Tuple[int, int], angle: float, max_range: int, angle_width: float, num_rays: int) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Perform ray-casting to simulate sonar data."""
    rows, cols = binary_map.shape
    sonar_data: List[Tuple[int, int]] = []
    theta: List[float] = []

    strong_signal = config.get('ray_cast', 'strong_signal')
    medium_signal = config.get('ray_cast', 'medium_signal')

    for i in range(num_rays):
        ray_angle = angle - (angle_width / 2) + (angle_width * i / num_rays)
        ray_angle_rad = np.radians(ray_angle)
        theta.append(ray_angle_rad)

        for ray_distance in range(max_range):
            y = int(pos[0] + ray_distance * np.cos(ray_angle_rad))
            x = int(pos[1] + ray_distance * np.sin(ray_angle_rad))
            if x < 0 or x >= cols or y < 0 or y >= rows:
                sonar_data.append((ray_distance, 0))  # No detection gives weaker signal
                break
            if binary_map[y, x] >= strong_signal:
                sonar_data.append((ray_distance, 2))
                break
            if binary_map[y, x] >= medium_signal:
                sonar_data.append((ray_distance, 1))  
                break 
        else:
            sonar_data.append((max_range, 0))  # Max range without hit
    
    return sonar_data, theta

def plot_both_views(room: np.ndarray, pos: Tuple[int, int], sonar_data: List[Tuple[int, int]], angle: float, angle_width: float, max_range: int, theta: List[float]) -> None:
    """ Plot both room view and sonar image view as a cone in polar coordinates. """
    plot_size = config.get('plotting', 'plot_size')
    room_color = config.get('plotting', 'room_color')
    sonar_position_color = config.get('plotting', 'sonar_position_color')
    sonar_ray_color = config.get('plotting', 'sonar_ray_color')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_size)

    # Plot for traditional room view
    ax1.imshow(room, cmap=room_color, origin='lower', interpolation='bilinear')
    ax1.scatter([pos[1]], [pos[0]], color=sonar_position_color)
    num_rays = len(sonar_data)
    for (r, strength), t in zip(sonar_data, theta):
        x = pos[0] + r * np.cos(t)
        y = pos[1] + r * np.sin(t)
        ax1.plot([pos[1], y], [pos[0], x], sonar_ray_color)
    ax1.set_title('Room with Pipe and Ground')

    relative_theta = [t - np.radians(angle) for t in theta]

    ax2 = plt.subplot(122, projection='polar')
    ax2.set_theta_zero_location('S')
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, max_range)
    ax2.set_xlim(-np.radians(angle_width / 2), np.radians(angle_width / 2))
    ax2.set_title('Sonar Image')
    ax2.set_facecolor('white')

    colors = viridis(np.linspace(0, 1, max_range))
    for (r, strength), t in zip(sonar_data, relative_theta):
        color = colors[int(r * strength)]
        ax2.scatter(t, r, color=color, s=10 * strength)

    plt.show()

def run_ideal_basic_sonar_simulation(sonar_position: Tuple[int, int], sonar_angle: float) -> None:
    dimensions = config.dimensions
    pipe_center = config.pipe_center
    pipe_radius = config.pipe_radius
    max_range = config.get('sonar', 'max_range')
    angle_width = config.get('sonar', 'angle_width')
    num_rays = config.get('sonar', 'num_rays')

    # Create room with pipe and ground wave
    room = create_room_with_pipe_and_ground(dimensions, pipe_center, pipe_radius, ground_wave_function)

    # Perform ray-casting
    sonar_data, theta = ray_cast(room, sonar_position, sonar_angle, max_range, angle_width, num_rays)

    # Visualize both views
    plot_both_views(room, sonar_position, sonar_data, sonar_angle, angle_width, max_range, theta)
