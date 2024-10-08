from ideal_simulation.basic_sonar import run_ideal_basic_sonar_simulation
from ideal_simulation.multiple_sonar import run_ideal_multiple_sonar_simulation
from ideal_simulation.terrain_sonar_scan import run_ideal_mesh_sonar_scan_simulation
from ideal_simulation.pipeline_seafloor_analysis import run_pipeline_seafloor_detection
from ideal_simulation.world_mapping import run_3d_mapping_simulation
from utils.ideal_simulation_utils import run_detection_evaluation
from config import config

import cProfile
import pstats
from contextlib import contextmanager

@contextmanager
def profiled():
    profiler = cProfile.Profile()
    profiler.enable()
    yield profiler
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(10)  # Print top 10 functions


        
def main():
    # with profiled():
    # ************ Simulation Parameters ************ 
    sonar_positions_1 =  [ (3000, 1000), (3000, 5000)] # (y, x)
    
    slice_position = 10
    slice_positions = config.get('mesh_processing', 'slice_positions')
    
    angles = [145,215]  # direction in degrees (mid-point direction pointing down)
    
    # ************ Run Basic Simulation ************
    # run_ideal_basic_sonar_simulation(sonar_positions_2[0], angles[0])
    
    # ************ Run Multiple Sonar Simulation ************
    # run_ideal_multiple_sonar_simulation(sonar_positions_1, angles)
    
    # ************ Run Mesh Sonar Simulation ************
    # run_ideal_mesh_sonar_scan_simulation(slice_position, sonar_positions_1, angles)
    
    # ************ Run Sonar Simulation ************
    # run_pipeline_seafloor_detection(slice_position, sonar_positions_1, angles, get_ground_truth=True, use_clustering=False)
    
    # ************ Run Detection Evaluation ************
    run_detection_evaluation(sonar_positions_1, angles, slice_positions)
    
    
    
    
if __name__ == "__main__":
    main()