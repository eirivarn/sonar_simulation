import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from ideal_simulation.basic_sonar import create_room_with_pipe_and_ground, ground_wave_function, ray_cast

def transform_to_global(pos, sonar_data, theta):
    """ Transform intersections from sonar back to the global coordinate system. """
    global_coords = []

    # Loop through sonar data and corresponding angles
    for (r, strength), t in zip(sonar_data, theta):
        # Calculate global x coordinate
        x = pos[0] + r * np.cos(t)
        # Calculate global y coordinate
        y = pos[1] + r * np.sin(t)
        # Append the calculated global coordinates and strength
        global_coords.append((x, y, strength))

    return global_coords

def transform_to_reference_sonar(ref_pos, ref_angle, global_coords):
    """ Transform global coordinates to the reference sonar's coordinate system. """
    transformed_coords = []

    # Convert reference angle to radians
    ref_angle_rad = np.radians(ref_angle)
    # Precompute cosine and sine of the negative reference angle
    cos_angle = np.cos(-ref_angle_rad)
    sin_angle = np.sin(-ref_angle_rad)

    # Loop through global coordinates
    for (x, y, strength) in global_coords:
        # Calculate the difference in x and y coordinates relative to the reference position
        dx = x - ref_pos[0]
        dy = y - ref_pos[1]
        # Rotate the coordinates to align with the reference angle
        transformed_x = dx * cos_angle - dy * sin_angle
        transformed_y = dx * sin_angle + dy * cos_angle
        # Calculate the range and angle in the reference coordinate system
        transformed_r = np.sqrt(transformed_x**2 + transformed_y**2)
        transformed_theta = np.arctan2(transformed_y, transformed_x)
        # Append the transformed coordinates and strength
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

def run_ideal_multiple_sonar_simulation(dimensions, pipe_center, pipe_radius, sonar_positions, angles, max_range, angle_width, num_rays):
    """ Run a basic sonar simulation with given parameters. """
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