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

def ground_wave_function(y, amplitude=10, frequency=0.05):
    """ Function to generate a wave-like ground using Perlin noise with higher resolution. """
    return int(15 + amplitude * pnoise1(y * frequency, repeat=1024))

def ray_cast(room, pos, angle, max_range, angle_width, num_rays):
    """ Perform ray-casting to simulate sonar data. """
    rows, cols = room.shape
    sonar_data = []
    theta = []

    for i in range(num_rays):
        ray_angle = angle - (angle_width / 2) + (angle_width * i / num_rays)
        ray_angle_rad = np.radians(ray_angle)
        theta.append(ray_angle_rad)

        for r in range(max_range):
            x = int(pos[0] + r * np.cos(ray_angle_rad))
            y = int(pos[1] + r * np.sin(ray_angle_rad))
            if x < 0 or x >= rows or y < 0 or y >= cols:
                sonar_data.append((r, 0))  # No detection gives weaker signal
                break
            if room[x, y] >= 0.5:
                sonar_data.append((r, 1))  # Detection gives stronger signal
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
        ax2.scatter(t, r, color=color, s=10 * strength + 1)

    plt.show()

"""
# Define room dimensions
dimensions = (1000, 1000)

# Define pipe parameters (circle)
pipe_center = (30, 500)  # (y, x)
pipe_radius = 50  # Radius of the pipe

# Define sonar parameters
pos = (500, 500)
angle = 180  # direction in degrees (mid-point direction pointing down)
max_range = 600
angle_width = 60  # total sonar angle width in degrees
num_rays = 100  # number of rays for higher resolution


# Create room with pipe and ground wave
room = create_room_with_pipe_and_ground(dimensions, pipe_center, pipe_radius, ground_wave_function)

# Perform ray-casting
sonar_data, theta = ray_cast(room, pos, angle, max_range, angle_width, num_rays)

# Visualize both views
plot_both_views(room, pos, sonar_data, angle, angle_width, max_range, theta)
"""