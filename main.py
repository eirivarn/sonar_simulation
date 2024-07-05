from ideal_simulation.basic_sonar import run_ideal_basic_sonar_simulation
from ideal_simulation.multiple_sonar import run_ideal_multiple_sonar_simulation
from ideal_simulation.terrain_sonar_scan import run_ideal_mesh_sonar_scan_simulation
from ideal_simulation.pipeline_seafloor_analysis import run_pipeline_seafloor_detection
#from ideal_simulation.detection_evaluation import * 


def main():
    # ************ Simulation Parameters ************
    pipe_center = (100, 500)  # center of the pipe (y, x)
    pipe_radius = 50  
    simulation_dimensions = (500, 1000)   # (y, x)  
    sonar_positions_1 =  [(1000, 1200)] # (y, x)
    sonar_positions_2 =  [(300, 400), (300, 600)] # (y, x)
    

    angles = [200]  # direction in degrees (mid-point direction pointing down)
    max_range = 1000
    angle_width = 90
    num_rays = 100  

    combined_mesh_path = ['/Users/eirikvarnes/code/blender/combined.obj']
    seperate_mesh_paths = ['/Users/eirikvarnes/code/blender/seafloor.obj', '/Users/eirikvarnes/code/blender/pipeline.obj']
    slice_position = -30
    slice_positions = list(range(-25, 25, 5))
    
    # Clustering parameters
    clustering_params_signal = {
    'DBSCAN': {'eps': 20, 'min_samples': 5},
    'KMeans': {'n_clusters': 3, 'random_state': 42},
    'Agglomerative': {'n_clusters': 3}}
    
    clustering_params_real = {
    'DBSCAN': {'eps': 2, 'min_samples': 12},
    'KMeans': {'n_clusters': 1, 'random_state': 42},
    'Agglomerative': {'n_clusters': 1}}

    # ************ Run Basic Simulation ************
    # run_ideal_basic_sonar_simulation(simulation_dimensions, pipe_center, pipe_radius, sonar_positions_2[0], angles[0], max_range, angle_width, num_rays)
    
    # ************ Run Multiple Sonar Simulation ************
    # run_ideal_multiple_sonar_simulation(simulation_dimensions, pipe_center, pipe_radius, sonar_positions_2, angles, max_range, angle_width, num_rays)
    
    # ************ Run Mesh Sonar Simulation ************
    # run_ideal_mesh_sonar_scan_simulation(seperate_mesh_paths, 'x', slice_position, sonar_positions, angles, max_range, angle_width, num_rays)
    
    # ************ Run Sonar Simulation with Clustering ************
    run_pipeline_seafloor_detection(seperate_mesh_paths, slice_position, sonar_positions_1, angles, max_range, angle_width, num_rays, clustering_params_signal, get_ground_truth=True, clustering_params_real=clustering_params_real)
    
    
    
if __name__ == "__main__":
    main()