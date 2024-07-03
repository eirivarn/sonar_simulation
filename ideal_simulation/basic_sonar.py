import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from noise import pnoise1

def create_room_with_pipe_and_ground(dimensions, pipe_center, pipe_radius, ground_wave):
    """ Create a 2D room with a pipe and ground wave. """
    room = np.zeros(dimensions)

    # Draw the pipe (circle) using sub-pixel rendering for smooth edges
    y, x = np.ogrid[:dimensions[0], :dimensions[1]]
    distance_from_center = np.sqrt((x - pipe_center[1])**2 + (y - pipe_center[0])**2)
    room += np.clip(1 - (distance_from_center - pipe_radius), 0, 1)

    # Draw the ground wave with smooth interpolation
    for y in range(dimensions[1]):
        x = ground_wave(y)
        if 0 <= x < dimensions[0]:
            room[x, y] = 1

    return room

def ground_wave_function(y, amplitude=2, frequency=0.05):
    """ Function to generate a wave-like ground using Perlin noise with higher resolution. """
    return int(45 + amplitude * pnoise1(y * frequency, repeat=1024))

def ray_cast(binary_map, pos, angle, max_range, angle_width, num_rays):
    """Perform ray-casting to simulate sonar data."""
    rows, cols = binary_map.shape
    sonar_data = []
    theta = []
    
    plt.figure(figsize=(8, 8))
    plt.imshow(binary_map, cmap='gray', origin='lower')
    plt.colorbar(label='Obstacle Presence (1 = Obstacle, 0 = Free Space)')
    plt.title('Binary Map Visualization')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()

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
            if binary_map[y, x] >= 0.5:
                sonar_data.append((ray_distance, 1))  # Detection gives stronger signal
                break
        else:
            sonar_data.append((max_range, 0))  # Max range without hit
    
    return sonar_data, theta

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
    ax1.set_title('Room with Pipe and Ground')

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
        ax2.scatter(t, r, color=color, s=10 * strength)

    plt.show()

def run_ideal_basic_sonar_simulation(dimensions, pipe_center, pipe_radius, pos, angle, max_range, angle_width, num_rays):
    """ Run a basic sonar simulation with given parameters. """
    # Create room with pipe and ground wave
    room = create_room_with_pipe_and_ground(dimensions, pipe_center, pipe_radius, ground_wave_function)
    y_range = (0, dimensions[0])
    z_range = (0, dimensions[1])
    # Perform ray-casting
    sonar_data, theta = ray_cast(room, pos, angle, max_range, angle_width, num_rays, )

    # Visualize both views
    plot_both_views(room, pos, sonar_data, angle, angle_width, max_range, theta)
    