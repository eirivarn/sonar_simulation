from typing import Tuple, List, Dict, Any
import numpy as np

class Config:
    def __init__(self):
        self.dimensions: Tuple[int, int] = (2000, 5000)
        self.area_dimensions: Tuple[int, int] = (35, 1000)
        self.pipe_center: Tuple[int, int] = (200, 2500)
        self.pipe_radius: int = 100
        self.combined_mesh_path: List[str] = ['/Users/eirikvarnes/code/blender/combined.obj']
        self.separate_mesh_paths: List[str] = ['/Users/eirikvarnes/code/blender/seafloor_to_scale.obj', '/Users/eirikvarnes/code/blender/pipeline_to_scale.obj']
        self.show_plots: bool = False
        self.get_ground_truth: bool = True
        self.load_data: bool = False  
        
        self.sonar: Dict[str, Any] = {
            "max_range": 5000,
            "angle_width": 45,
            "num_rays": 90,
            "sonar_positions": [(50, 20), (30, 40)],
            "angles": [90, 45],
        }
        
        self.clustering_params: Dict[str, Dict[str, Any]] = {
            "ransac": {
                "min_samples": 1,
                "residual_threshold": 90,
                "max_trials": 1000
            },
            "dbscan": {
                "eps": 10,
                "min_samples": 1,
            },
        }

        self.plotting: Dict[str, Any] = {
            "room_color": 'gray',
            "sonar_position_color": 'red',
            "sonar_ray_color": 'yellow',
            "plot_size": [14, 6],
            "image_size": (15, 15),
            "scatter_point_size": 50,
            "alpha": 0.5,
            "pipe_color_map": 'viridis',
            "seabed_color_map": 'terrain'
        }
        self.assessment: Dict[str, Any] = {
            "angle_weight": 0.3,
            "area_weight": 0.3,
            "distance_weight": 0.4,
            "free_span_threshold": 0.1
        }
        self.mesh_processing: Dict[str, Any] = {
            "slice_axis": 'x',
            "slice_axes": ['x', 'y', 'z'],
            "padding_factor": 5,
            "grid_size": (700, 700),
            "slice_positions": list(range(-90, 90, 40)),
            "rotation_matrix": [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        }
        self.interpolation: Dict[str, Any] = {
            "num_bins": 500,
            "smoothing_factor": 20.0,
            "curve_outlier_threshold": 10.0,
            "circle_point_margin": 15.0,
            "buffer_distance": 5.0  
        }
        self.ray_cast: Dict[str, Any] = {
            "strong_signal": 2,
            "medium_signal": 1,
            "no_signal": 0
        }
        self.ground_wave = {
            "base_level": 100,  # Middle of vertical range
            "components": [
                {"amplitude": 75, "frequency": 0.005, "phase_shift": np.pi / 4},  # Very large wavelength, large amplitude
                #{"amplitude": 20, "frequency": 0.005, "phase_shift": np.pi / 4},  # Large wavelength, medium amplitude
                #{"amplitude": 12, "frequency": 0.02, "phase_shift": np.pi / 2},   # Smaller wavelength, relatively large amplitude
                #{"amplitude": 6, "frequency": 0.03, "phase_shift": np.pi}         # Even smaller wavelength
            ]
        }


    def get(self, section: str, key: str) -> Any:
        return getattr(self, section).get(key)

# Initialize the config object
config = Config()
