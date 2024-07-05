from ideal_simulation.basic_sonar import run_ideal_basic_sonar_simulation
from ideal_simulation.multiple_sonar import run_ideal_multiple_sonar_simulation
from ideal_simulation.terrain_sonar_scan import run_ideal_mesh_sonar_scan_simulation
from ideal_simulation.pipeline_seafloor_analysis import run_pipeline_seafloor_detection
#from ideal_simulation.detection_evaluation import * 


def main():
    # ************ Simulation Parameters ************ 
    sonar_positions_1 =  [(1000, 1200)] # (y, x)
    sonar_positions_2 =  [(300, 400), (300, 700)] # (y, x)
    

    angles = [180, 190]  # direction in degrees (mid-point direction pointing down)


    combined_mesh_path = ['/Users/eirikvarnes/code/blender/combined.obj']
    seperate_mesh_paths = ['/Users/eirikvarnes/code/blender/seafloor.obj', '/Users/eirikvarnes/code/blender/pipeline.obj']
    slice_position = -30
    slice_positions = list(range(-25, 25, 5))

    # ************ Run Basic Simulation ************
    run_ideal_basic_sonar_simulation(sonar_positions_2[0], angles[0])
    
    # ************ Run Multiple Sonar Simulation ************
    run_ideal_multiple_sonar_simulation(sonar_positions_2, angles)
    
    # ************ Run Mesh Sonar Simulation ************
    # run_ideal_mesh_sonar_scan_simulation(seperate_mesh_paths, 'x', slice_position, sonar_positions, angles, max_range, angle_width, num_rays)
    
    # ************ Run Sonar Simulation with Clustering ************
    # run_pipeline_seafloor_detection(seperate_mesh_paths, slice_position, sonar_positions_1, angles, max_range, angle_width, num_rays, clustering_params_signal, get_ground_truth=True, clustering_params_real=clustering_params_real)
    
    
    
if __name__ == "__main__":
    main()