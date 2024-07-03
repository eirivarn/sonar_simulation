from ideal_simulation.basic_sonar import run_ideal_basic_sonar_simulation
from ideal_simulation.multiple_sonar import run_ideal_multiple_sonar_simulation
from ideal_simulation.terrain_sonar_scan import run_ideal_mesh_sonar_scan_simulation
from ideal_simulation.terrain_sonar_mapping import run_ideal_mesh_sonar_mapping_simulation
from ideal_simulation.pipeline_seafloor_analysis import run_pipeline_seafloor_detection
#from ideal_simulation.detection_evaluation import * 


def main():
    # ************ Simulation Parameters ************
    pipe_center = (100, 500)  # center of the pipe (y, x)
    pipe_radius = 50  
    simulation_dimensions = (1000, 1000)   # (y, x)  
    sonar_positions = [(300, 350), (300, 500), (300, 650)] # (y, x)

    angles = [180, 180, 180]  # direction in degrees (mid-point direction pointing down)
    max_range = 500
    angle_width = 60  # total sonar angle width in degrees
    num_rays = 5  # number of rays for higher resolution

    mesh_paths = ['/Users/eirikvarnes/code/totalenergies/simulation_test/blender_terrain_test_1.obj']
    seperate_mesh_paths = ['/Users/eirikvarnes/code/blender/pipeline.obj', '/Users/eirikvarnes/code/blender/seafloor.obj']
    slice_position = -10
    slice_positions = list(range(-25, 25, 5))
    
    # Clustering parameters
    clustering_params = {
    'DBSCAN': {'eps': 16, 'min_samples': 5},
    'KMeans': {'n_clusters': 2, 'random_state': 42},
    'Agglomerative': {'n_clusters': 2}}

    # ************ Run Basic Simulation ************
    run_ideal_basic_sonar_simulation(simulation_dimensions, pipe_center, pipe_radius, sonar_positions[1], angles[1], max_range, angle_width, num_rays)
    
    # ************ Run Multiple Sonar Simulation ************
    run_ideal_multiple_sonar_simulation(simulation_dimensions, pipe_center, pipe_radius, sonar_positions, angles, max_range, angle_width, num_rays)
    
    # ************ Run Mesh Sonar Simulation ************
    run_ideal_mesh_sonar_scan_simulation(seperate_mesh_paths, 'x', slice_position, sonar_positions, angles, max_range, angle_width, num_rays)
    
    # ************ Run Mesh Sonar Mapping Simulation ************
    # run_ideal_mesh_sonar_mapping_simulation(mesh_path, simulation_dimensions, 'x', slice_positions, sonar_positions, angles, max_range, angle_width, num_rays)
    
    # ************ Create Sonar Image from Ray Cast ************
    # save_sonar_image(mesh_path, slice_position, simulation_dimensions, sonar_positions, angles, max_range, angle_width, num_rays)
    
    # ************ Run Sonar Simulation with Clustering ************
    # run_pipeline_seafloor_detection(mesh_path, slice_position, simulation_dimensions, sonar_positions, angles, max_range, angle_width, num_rays, clustering_params)

    
if __name__ == "__main__":
    main()
