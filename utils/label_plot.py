import matplotlib.pyplot as plt
import numpy as np

# Define the circle and the radius
radius = 1
circle = plt.Circle((0, 0), radius, color='blue', fill=False, linewidth=2)

# Create figure and axis
fig, ax = plt.subplots()

# Add circle to plot
ax.add_artist(circle)

# Define labels and their positions
label_positions = np.linspace(-11, 11, num=23)  # Positions from -11 to 11 inclusive
x_positions = label_positions * radius
y_positions = np.zeros_like(x_positions)

# Plot points with labels
for x, label in zip(x_positions, label_positions):
    ax.plot(x, 0, 'o', label=f'Label {int(label)}')
    ax.text(x, 0.1, f'{int(label)}', horizontalalignment='center')

# Set limits, aspect
ax.set_xlim(-12, 12)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)

# Add labels and title
ax.set_title('Circle with Labeled Points')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')

# Show plot with legend
ax.legend(loc='upper right')
plt.show()