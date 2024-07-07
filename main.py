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

    # ************ Run Basic Simulation ************
    # run_ideal_basic_sonar_simulation(sonar_positions_2[0], angles[0])
    
    # ************ Run Multiple Sonar Simulation ************
    # run_ideal_multiple_sonar_simulation(sonar_positions_2, angles)
    
    # ************ Run Mesh Sonar Simulation ************
    # run_ideal_mesh_sonar_scan_simulation(sonar_positions_1, angles)
    
    # ************ Run Sonar Simulation with Clustering ************
    run_pipeline_seafloor_detection(sonar_positions_1, angles, get_ground_truth=True)
    
    
    
if __name__ == "__main__":
    main()