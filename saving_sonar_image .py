from realistic_terrain_sonar_scann import * 

def create_sonar_image(sonar_data, max_range, num_rays):
  """ Create a sonar image from the ray cast data. """
  image = np.zeros((max_range, num_rays))  # Initialize image with zeros
  for i, ray_data in enumerate(sonar_data):
    for r, strength in ray_data:
      if 0 <= r < max_range:
        image[r, i] = strength

  # Apply smoothing (optional)
  image = cv2.GaussianBlur(image, (5, 5), 0)
  image = cv2.resize(image, (400, 500), interpolation=cv2.INTER_LINEAR)
  return image


position = 10
filename = f"sonar_results/position_{position}.png"
dimensions = (1000, 1000)
sonar_position = (700, 500)
angle = 180
max_range = 700
angle_width = 45
num_rays = 50

slice_df = extract_2d_slice_from_mesh(terrain, position, axis='x')

if slice_df is not None:
    binary_map = create_binary_map_from_slice(dimensions, slice_df)
    

    # Perform ray-casting on the binary map
    sonar_data, theta = ray_cast(binary_map, sonar_position, angle, max_range, angle_width, num_rays)

    # Visualize both views
    # plot_both_views(binary_map, sonar_position, sonar_data, angle, angle_width, max_range, theta)
    
    # Create sonar image from hits
    sonar_image = create_sonar_image(sonar_data, max_range, num_rays)  # Adjust max_range

    # Save the sonar image
    filename = f"sonar_results/position_{position}.png"
    plt.imsave(filename, sonar_image)
else:
    print("No slice data available to display.")