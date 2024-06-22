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

def transform_to_global(pos, sonar_data, theta):
    """ Transform intersections from sonar back to the global coordinate system. """
    global_coords = []
    for (r, strength), t in zip(sonar_data, theta):
        x = pos[0] + r * np.cos(t)
        y = pos[1] + r * np.sin(t)
        global_coords.append((x, y, strength))
    return global_coords

def transform_to_reference_sonar(ref_pos, ref_angle, global_coords):
    """ Transform global coordinates to the reference sonar's coordinate system. """
    transformed_coords = []
    ref_angle_rad = np.radians(ref_angle)
    cos_angle = np.cos(-ref_angle_rad)
    sin_angle = np.sin(-ref_angle_rad)
    for (x, y, strength) in global_coords:
        dx = x - ref_pos[0]
        dy = y - ref_pos[1]
        transformed_x = dx * cos_angle - dy * sin_angle
        transformed_y = dx * sin_angle + dy * cos_angle
        transformed_r = np.sqrt(transformed_x**2 + transformed_y**2)
        transformed_theta = np.arctan2(transformed_y, transformed_x)
        transformed_coords.append((transformed_r, transformed_theta, strength))
    return transformed_coords

def plot_both_views(room, sonar_positions, all_sonar_data, angles, angle_width, max_range, all_theta):
    """ Plot both room view and sonar image view as a cone in polar coordinates. """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for traditional room view
    ax1.imshow(room, cmap='gray', origin='lower', interpolation='bilinear')
    colors = ['red', 'blue', 'green', 'yellow']
    
    for idx, (pos, sonar_data, theta) in enumerate(zip(sonar_positions, all_sonar_data, all_theta)):
        ax1.scatter([pos[1]], [pos[0]], color=colors[idx % len(colors)])  # Sonar position
        for (r, strength), t in zip(sonar_data, theta):
            x = pos[0] + r * np.cos(t)
            y = pos[1] + r * np.sin(t)
            ax1.plot([pos[1], y], [pos[0], x], color=colors[idx % len(colors)])
    
    ax1.set_title('Room with Pipe and Ground')

    # Transform all sonar data to global coordinates
    global_coords = []
    for pos, sonar_data, theta in zip(sonar_positions, all_sonar_data, all_theta):
        global_coords.extend(transform_to_global(pos, sonar_data, theta))

    # Use the first sonar as the reference sonar
    ref_pos = sonar_positions[1]
    ref_angle = angles[1]

    # Transform all global coordinates to the reference sonar's coordinate system
    transformed_coords = transform_to_reference_sonar(ref_pos, ref_angle, global_coords)

    # Plot for sonar image view as a cone
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_theta_zero_location('S')  # Set zero angle to the top (straight up)
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, max_range)
    ax2.set_xlim(-np.radians(angle_width / 2), np.radians(angle_width / 2))  # Center the sonar field of view
    ax2.set_title('Sonar Image')
    ax2.set_facecolor('white')

    colors = viridis(np.linspace(0, 1, max_range))
    for (r, t, strength) in transformed_coords:
        if -np.radians(angle_width / 2) <= t <= np.radians(angle_width / 2):
            color = colors[int(r * strength)]
            ax2.scatter(t, r, color=color, s=10 * strength + 1)

    plt.show()

# Define room dimensions
dimensions = (1000, 1000)

# Define pipe parameters (circle)
pipe_center = (30, 500)  # (y, x)
pipe_radius = 50  # Radius of the pipe

# Define sonar parameters
sonar_positions = [(250, 250), (500, 500), (250, 750)]  # Add more sonar positions as needed
angles = [120, 180, 230]  # directions in degrees (mid-point direction pointing down)
max_range = 1000
angle_width = 60  # total sonar angle width in degrees
num_rays = 100  # number of rays for higher resolution

# Create room with pipe and ground wave
room = create_room_with_pipe_and_ground(dimensions, pipe_center, pipe_radius, ground_wave_function)

# Perform ray-casting for each sonar
all_sonar_data = []
all_theta = []

for pos, angle in zip(sonar_positions, angles):
    sonar_data, theta = ray_cast(room, pos, angle, max_range, angle_width, num_rays)
    all_sonar_data.append(sonar_data)
    all_theta.append(theta)

# Visualize both views
plot_both_views(room, sonar_positions, all_sonar_data, angles, angle_width, max_range, all_theta)
