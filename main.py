from ideal_simulation.basic_sonar import run_ideal_basic_sonar_simulation
from ideal_simulation.multiple_sonar import run_ideal_multiple_sonar_simulation
from ideal_simulation.terrain_sonar_scan import run_ideal_mesh_sonar_scan_simulation
from ideal_simulation.pipeline_seafloor_analysis import run_pipeline_seafloor_detection
from ideal_simulation.world_mapping import run_3d_mapping_simulation
from ideal_simulation.save_results import save_results_to_csv, format_filename
from config import config

def main():
    # ************ Simulation Parameters ************ 
    sonar_positions_1 =  [(1500, 250)] # (y, x)
    
    slice_position = -50
    slice_positions = config.get('mesh_processing', 'slice_positions')
    
    angles = [150]  # direction in degrees (mid-point direction pointing down)x

    # ************ Run Basic Simulation ************
    # run_ideal_basic_sonar_simulation(sonar_positions_2[0], angles[0])
    
    # ************ Run Multiple Sonar Simulation ************
    # run_ideal_multiple_sonar_simulation(sonar_positions_2, angles)
    
    # ************ Run Mesh Sonar Simulation ************
    # run_ideal_mesh_sonar_scan_simulation(sonar_positions_1, angles)
    
    # ************ Run Sonar Simulation with Clustering ************
    run_pipeline_seafloor_detection(slice_position, sonar_positions_1, angles, get_ground_truth=True)
    
    # # ************ Run Detection Evaluation ************
    # results = run_3d_mapping_simulation(sonar_positions_1, angles, slice_positions)
    
    # signal_results = [r[:10] for r in results if r is not None and len(r) >= 10]
    # ground_truth_results = [r[10] for r in results if r is not None and len(r) > 10]
    
    # if results:
    #     signal_filename = format_filename('signal_results', sonar_positions_1, angles)
    #     save_results_to_csv(signal_filename, 'signal', signal_results)
        
    #     save_results_to_csv('data/ground_truth_results.csv', 'ground_truth', ground_truth_results)
    # else:
    #     print("No results to save.")
    
if __name__ == "__main__":
    main()