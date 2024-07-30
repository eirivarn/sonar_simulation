import pyvista as pv
import pandas as pd
import os

def convert_obj_to_csv(obj_path: str, csv_path: str):
    if not obj_path.lower().endswith('.obj'):
        raise ValueError("The file provided is not an OBJ file.")
    mesh = pv.read(obj_path)
    points = mesh.points
    df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def visualize_obj_and_csv(obj_path: str, csv_path: str):
    # Load the OBJ mesh
    mesh = pv.read(obj_path)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Convert DataFrame to PyVista PolyData
    cloud = pv.PolyData(df.values)

    # Set up the visualization
    plotter = pv.Plotter(notebook=False, shape=(1, 2))

    # Visualize the OBJ mesh
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, color='lightblue', show_edges=True, label='OBJ Mesh')
    plotter.add_legend()
    plotter.set_background('white')
    plotter.add_text("OBJ Mesh Visualization", font_size=10)

    # Visualize the CSV points
    plotter.subplot(0, 1)
    plotter.add_mesh(cloud, point_size=5, label='CSV Points')
    plotter.add_legend()
    plotter.set_background('white')
    plotter.add_text("CSV Points Visualization", font_size=10)

    # Show the plot
    plotter.show()

# Paths to your files
obj_path = '/Users/eirikvarnes/code/blender/seafloor_to_scale.obj'
csv_path = 'output_file.csv'

# Convert and visualize
convert_obj_to_csv(obj_path, csv_path)
visualize_obj_and_csv(obj_path, csv_path)