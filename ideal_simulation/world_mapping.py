import numpy as np
import os
from config import config
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from ideal_simulation.pipeline_seafloor_analysis import run_pipeline_seafloor_detection


def run_3d_seafloor_analysis(sonar_positions, angles):
    slice_positions = config.get('mesh_processing', 'slice_positions')
    results = []
    
    for slice_position in slice_positions:
        result = run_pipeline_seafloor_detection(slice_position, sonar_positions, angles)
            