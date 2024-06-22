import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise1
import random
import cv2

def create_room_with_pipe_ground_and_debris(dimensions, pipe_center, pipe_radius, ground_wave):
    """ Create a 2D room with a pipe, ground wave, and random debris. """
    room = np.zeros(dimensions)

    # Draw the pipe (circle)
    y, x = np.ogrid[:dimensions[0], :dimensions[1]]
    distance_from_center = np.sqrt((x - pipe_center[1])**2 + (y - pipe_center[0])**2)
    room += np.clip(1 - (distance_from_center - pipe_radius), 0, 1)

    # Draw the ground wave
    for y in range(dimensions[1]):
        x = ground_wave(y)
        if 0 <= x < dimensions[0]:
            room[x, y] = 1

    # Add random debris with varying shapes and reflectivity
    num_debris = 1000  # Increase the number of debris
    for _ in range(num_debris):
        shape_type = random.choice(['circle', 'rectangle', 'ellipse'])
        reflectivity = random.uniform(0.2, 0.6)  # Reflectivity range for weaker signals
        if shape_type == 'circle':
            center = (random.randint(0, dimensions[1] - 1), random.randint(0, dimensions[0] - 1))
            radius = random.randint(1, 3)
            cv2.circle(room, center, radius, reflectivity, -1)
        elif shape_type == 'ellipse':
            center = (random.randint(0, dimensions[1] - 1), random.randint(0, dimensions[0] - 1))
            axes = (random.randint(1, 3), random.randint(1, 3))
            angle = random.randint(0, 180)
            cv2.ellipse(room, center, axes, angle, 0, 360, reflectivity, -1)

    # Apply Gaussian blur to the room
    # room = cv2.GaussianBlur(room, (15, 15), 0)

    return room

def ground_wave_function(y, amplitude=10, frequency=0.05):
    """ Function to generate a wave-like ground using Perlin noise with higher resolution. """
    return int(15 + amplitude * pnoise1(y * frequency, repeat=1024))

def material_reflectivity(material_value):
    """ Determine reflectivity based on material value. """
    if material_value > 0.8:
        return 0.75  # Strong reflector (e.g., metal)
    elif material_value > 0.5:
        return 0.33  # Moderate reflector (e.g., debris)
    else:
        return 0.15  # Weak reflector (e.g., sediment)

def calculate_multipath_reflections(material_value, incident_strength):
    """ Calculate reflections and transmissions based on material reflectivity. """
    reflectivity = material_reflectivity(material_value)
    reflected_strength = incident_strength * reflectivity
    transmitted_strength = incident_strength * (1 - reflectivity)
    return reflected_strength, transmitted_strength

def ray_cast(room, pos, angle, max_range, angle_width, num_rays, attenuation_factor=0.005):
    """ Perform ray-casting to simulate sonar data with multipath reflections and water attenuation. """
    rows, cols = room.shape
    sonar_data = []
    theta = []

    for i in range(num_rays):
        ray_angle = angle - (angle_width / 2) + (angle_width * i / num_rays)
        ray_angle_rad = np.radians(ray_angle)
        theta.append(ray_angle_rad)
        incident_strength = 1.0

        reflections = []
        for r in range(max_range):
            # Apply attenuation due to water
            current_strength = incident_strength * np.exp(-attenuation_factor * r)*0.8
            x = int(pos[0] + r * np.cos(ray_angle_rad))
            y = int(pos[1] + r * np.sin(ray_angle_rad))
            if x < 0 or x >= rows or y < 0 or y >= cols:
                reflections.append((r, current_strength))  # No detection gives weaker signal
                break
            if room[x, y] >= 0.5:
                reflected_strength, transmitted_strength = calculate_multipath_reflections(room[x, y], current_strength)
                reflections.append((r, reflected_strength))  # Detection gives reflected signal
                incident_strength = transmitted_strength  # Continue with transmitted signal
                if incident_strength < 0.1:  # Stop if the transmitted signal is too weak
                    break

        # Add noise and distortion to reflections
        distorted_reflections = [(int(r + np.random.normal(0, 2)), strength * np.random.uniform(0.9, 1.1)) for r, strength in reflections]
        sonar_data.append(distorted_reflections if distorted_reflections else [(max_range, 0)])  # Max range without hit

    return sonar_data, theta

def plot_both_views(room, pos, sonar_data, angle, angle_width, max_range, theta):
    """ Plot both room view and sonar image view as a cone in polar coordinates. """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for traditional room view
    ax1.imshow(room, cmap='turbo', origin='lower', interpolation='bilinear')  # Change colormap to 'plasma'
    ax1.scatter([pos[1]], [pos[0]], color='red')  # Sonar position
    for reflections, t in zip(sonar_data, theta):
        for r, strength in reflections:
            x = pos[0] + r * np.cos(t)
            y = pos[1] + r * np.sin(t)
            ax1.plot([pos[1], y], [pos[0], x], 'r-', alpha=0.5, linewidth=0.5)  # Plot the ray path
    ax1.set_title('Room with Pipe, Ground, and Debris')

    # Calculate relative angles to sonar
    # relative_theta = [t - np.radians(angle) for t in theta]

    # Create a 2D array to store signal strengths for the sonar view
    signal_grid = np.zeros((max_range, len(theta)))

    for i, (reflections, t) in enumerate(zip(sonar_data, theta)):
        for r, strength in reflections:
            if 0 <= r < max_range:
                signal_grid[r, i] = strength*50

    # Smooth the signal grid
    signal_grid = cv2.GaussianBlur(signal_grid, (5, 5), 0)

    # Plot for sonar image view as a cone
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_theta_zero_location('S')  # Set zero angle to the top (straight up)
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, max_range)
    ax2.set_xlim(-np.radians(angle_width / 2), np.radians(angle_width / 2))  # Center the sonar field of view
    ax2.set_title('Sonar Image')
    ax2.set_facecolor('white')

    # Use imshow to plot the signal grid
    extent = [-np.radians(angle_width / 2), np.radians(angle_width / 2), 0, max_range]
    ax2.imshow(signal_grid, aspect='auto', extent=extent, origin='lower', cmap='turbo', alpha=1)

    plt.show()

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

# Create room with pipe, ground wave, and random debris
room = create_room_with_pipe_ground_and_debris(dimensions, pipe_center, pipe_radius, ground_wave_function)

# Perform ray-casting
sonar_data, theta = ray_cast(room, pos, angle, max_range, angle_width, num_rays)

# Visualize both views
plot_both_views(room, pos, sonar_data, angle, angle_width, max_range, theta)
