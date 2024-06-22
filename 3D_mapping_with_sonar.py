import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from noise import pnoise1

def create_room_with_pipe_and_ground(dimensions, pipe_center, pipe_radius, ground_wave):
    """ Efficiently create a 2D room with a pipe and ground wave using vectorization. """
    y, x = np.ogrid[:dimensions[0], :dimensions[1]]
    room = ((x - pipe_center[1])**2 + (y - pipe_center[0])**2 <= pipe_radius**2).astype(int)
    
    # Adding ground wave
    y = np.arange(dimensions[1])
    x = ground_wave(y)
    valid = (x >= 0) & (x < dimensions[0])
    room[x[valid], y[valid]] = 1
    
    return room

def ground_wave_function(y, amplitude=10, frequency=0.05):
    return (15 + amplitude * np.sin(frequency * y)).astype(int)

def ray_cast(room, pos, angle, max_range, angle_width, num_rays):
    """ Perform ray-casting to simulate sonar data. """
    theta = np.radians(np.linspace(angle - angle_width / 2, angle + angle_width / 2, num_rays))
    r = np.arange(max_range)
    x = pos[0] + np.outer(r, np.cos(theta)).astype(int)
    y = pos[1] + np.outer(r, np.sin(theta)).astype(int)

    # Ensure indices are within the room boundaries
    valid = (x >= 0) & (x < room.shape[0]) & (y >= 0) & (y < room.shape[1])
    detections = np.zeros_like(x, dtype=bool)

    # Only update detections where valid
    detections[valid] = room[x[valid], y[valid]] == 1

    return x, y, detections

def plot_3d_sonar_images(room, angle, angle_width, max_range, num_positions):
    """ Plot 3D representation of sonar images along the pipe. """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for z in range(num_positions):
        pos = (50, 50)  # Assume moving along Z-axis does not change XY positions
        x, y, detections = ray_cast(room, pos, angle, max_range, angle_width, 100)

        # Flatten arrays for plotting
        x = x[detections]
        y = y[detections]

        # Plot each detection point in 3D
        ax.scatter(y, x, zs=z, c='b', marker='o')

    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z (Position along pipe)')
    ax.set_title('3D Representation of Sonar Images along Pipe')
    plt.show()

# Example parameters
angle = 180
max_range = 50
angle_width = 60
num_positions = 30

# Example room setup
dimensions = (100, 100)
pipe_center = (15, 50)
pipe_radius = 10
room = create_room_with_pipe_and_ground(dimensions, pipe_center, pipe_radius, ground_wave_function)

# Plotting the sonar scans in 3D
plot_3d_sonar_images(room, angle, angle_width, max_range, num_positions)
