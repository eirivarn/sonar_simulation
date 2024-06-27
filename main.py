from ideal_simulation.basic_sonar import run_ideal_basic_sonar_simulation
from ideal_simulation.multiple_sonar import run_ideal_multiple_sonar_simulation
from ideal_simulation.terrain_sonar_scann import run_ideal_mesh_sonar_scann_simulation
from ideal_simulation.terrain_sonar_mapping import run_ideal_mesh_sonar_mapping_simulation
from ideal_simulation.freespan_detection import run_sonar_simulation_with_clustering
from ideal_simulation.retriving_data_from_sonar import create_sonar_image

def main():
    # ************ Simulation Parameters ************
    simulation_dimensions = (1000, 1000)  # Dimensions of the simulation room

    pipe_center = (30, 500)  # (y, x)
    pipe_radius = 50  

    sonar_positions = [(250, 250), (500, 500), (250, 750)]
    
    angles = [110]  # direction in degrees (mid-point direction pointing down)
    max_range = 600
    angle_width = 60  # total sonar angle width in degrees
    num_rays = 100  # number of rays for higher resolution

    mesh_path = '/Users/eirikvarnes/code/totalenergies/simulation_test/blender_terrain_test_1.obj'
    slice_position = -0
    slice_positions = list(range(-25, 25, 5))
    
    # Clustering parameters
    eps = 1.5               # Maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples = 3         # The number of samples in a neighborhood for a point to be considered as a core point
    
    # ************ Run Basic Simulation ************
    # run_ideal_basic_sonar_simulation(simulation_dimensions, pipe_center, pipe_radius, sonar_positions[1], angles[1], max_range, angle_width, num_rays)
    
    # ************ Run Multiple Sonar Simulation ************
    # run_ideal_multiple_sonar_simulation(simulation_dimensions, pipe_center, pipe_radius, sonar_positions, angles, max_range, angle_width, num_rays)
    
    # ************ Run Mesh Sonar Simulation ************
    # run_ideal_mesh_sonar_scann_simulation(mesh_path, simulation_dimensions, 'x', slice_position, sonar_positions, angles, max_range, angle_width, num_rays)
    
    # ************ Run Mesh Sonar Mapping Simulation ************
    # run_ideal_mesh_sonar_mapping_simulation(mesh_path, simulation_dimensions, 'x', slice_positions, sonar_positions, angles, max_range, angle_width, num_rays)
    
    # ************ Create Sonar Image from Ray Cast ************
    
    
    # ************ Run Sonar Simulation with Clustering ************
    # run_sonar_simulation_with_clustering(mesh_path, slice_position[0], simulation_dimensions, sonar_positions[0], angles[0], max_range, angle_width, num_rays, eps, min_samples)

    
if __name__ == "__main__":
    main()
