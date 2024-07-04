import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from ideal_simulation.basic_sonar import create_room_with_pipe_and_ground, ground_wave_function, ray_cast

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

def plot_both_views(world, y_range, z_range, sonar_positions, all_sonar_data, angles, angle_width, max_range, all_theta, plot=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(world, extent=(y_range[0], y_range[1], z_range[0], z_range[1]), cmap='gray', origin='lower', interpolation='bilinear')
    colors = ['red', 'blue', 'green', 'yellow']
    
    for idx, (pos, sonar_data, theta) in enumerate(zip(sonar_positions, all_sonar_data, all_theta)):
        ax1.scatter([pos[1]], [pos[0]], color=colors[idx % len(colors)])
        for (r, strength), t in zip(sonar_data, theta):
            y = pos[0] + r * np.cos(t)
            z = pos[1] + r * np.sin(t)
            ax1.plot([pos[1], z], [pos[0], y], color=colors[idx % len(colors)])
    
    ax1.set_title('Room with Pipe and Ground')
    ax1.set_xlim(y_range)
    ax1.set_ylim(z_range)

    global_coords = []
    for pos, sonar_data, theta in zip(sonar_positions, all_sonar_data, all_theta):
        global_coords.extend(transform_to_global(pos, sonar_data, theta))

    ref_pos = sonar_positions[0]
    ref_angle = angles[0]

    transformed_coords = transform_to_reference_sonar(ref_pos, ref_angle, global_coords)
    
    if plot:
        ax2 = plt.subplot(122, projection='polar')
        ax2.set_theta_zero_location('S')
        ax2.set_theta_direction(-1)
        ax2.set_ylim(0, max_range)
        ax2.set_xlim(-np.radians(angle_width / 2), np.radians(angle_width / 2))
        ax2.set_title('Sonar Image')
        ax2.set_facecolor('white')

        strengths = np.array([coord[2] for coord in transformed_coords])
        colors = viridis((strengths - np.min(strengths)) / (np.max(strengths) - np.min(strengths)+1))

        for (r, t, strength), color in zip(transformed_coords, colors):
            if -np.radians(angle_width / 2) <= t <= np.radians(angle_width / 2):
                ax2.scatter(t, r, color=color, s=50 * strength)

        plt.show()

    return transformed_coords

def run_ideal_multiple_sonar_simulation(dimensions, pipe_center, pipe_radius, sonar_positions, angles, max_range, angle_width, num_rays):
    room = create_room_with_pipe_and_ground(dimensions, pipe_center, pipe_radius, ground_wave_function)
    y_range = (0, dimensions[1])
    z_range = (0, dimensions[0])
    
    all_sonar_data = []
    all_theta = []

    for pos, angle in zip(sonar_positions, angles):
        sonar_data, theta = ray_cast(room, pos, angle, max_range, angle_width, num_rays)
        all_sonar_data.append(sonar_data)
        all_theta.append(theta)

    plot_both_views(room, y_range, z_range, sonar_positions, all_sonar_data, angles, angle_width, max_range, all_theta)
