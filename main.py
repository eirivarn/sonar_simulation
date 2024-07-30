from ideal_simulation.basic_sonar import run_ideal_basic_sonar_simulation
from ideal_simulation.multiple_sonar import run_ideal_multiple_sonar_simulation
from ideal_simulation.terrain_sonar_scan import run_ideal_mesh_sonar_scan_simulation
from ideal_simulation.pipeline_seafloor_analysis import run_pipeline_seafloor_detection
from ideal_simulation.world_mapping import run_3d_mapping_simulation
from utils.ideal_simulation_utils import run_detection_evaluation
from config import config

def main():
    # ************ Simulation Parameters ************ 
    sonar_positions_1 =  [(1000, 3500)] # (y, x)
    slice_position = 70
    slice_positions = config.get('mesh_processing', 'slice_positions')
    
    angles = [200]  # direction in degrees (mid-point direction pointing down)x

    # ************ Run Basic Simulation ************
    # run_ideal_basic_sonar_simulation(sonar_positions_2[0], angles[0])
    
    # ************ Run Multiple Sonar Simulation ************
    # run_ideal_multiple_sonar_simulation(sonar_positions_2, angles)
    
    # ************ Run Mesh Sonar Simulation ************
    # run_ideal_mesh_sonar_scan_simulation(sonar_positions_1, angles)
    
    # ************ Run Sonar Simulation ************
    run_pipeline_seafloor_detection(slice_position, sonar_positions_1, angles, get_ground_truth=False, use_clustering=False)
    
    # ************ Run Detection Evaluation ************
    # run_detection_evaluation(sonar_positions_1, angles, slice_positions)

    
if __name__ == "__main__":
    main()