from typing import Tuple, List, Dict, Any

class Config:
    def __init__(self):
        self.dimensions: Tuple[int, int] = (500, 1000)
        self.pipe_center: Tuple[int, int] = (90, 500)
        self.pipe_radius: int = 30
        self.combined_mesh_path: List[str] = ['/Users/eirikvarnes/code/blender/combined.obj']
        self.separate_mesh_paths: List[str] = ['/Users/eirikvarnes/code/blender/seafloor.obj', '/Users/eirikvarnes/code/blender/pipeline.obj']
        self.show_plots: bool = False
        
        self.sonar: Dict[str, Any] = {
            "max_range": 1300,
            "angle_width": 60,
            "num_rays": 120,
            "sonar_positions": [(50, 20), (30, 40)],
            "angles": [90, 45],
        }
        
        self.clustering_params: Dict[str, Dict[str, Any]] = {
            "kmeans": {
                "n_clusters": 3
            },
            "agglomerative": {
                "n_clusters": 3
            }
        }

        self.plotting: Dict[str, Any] = {
            "room_color": 'gray',
            "sonar_position_color": 'red',
            "sonar_ray_color": 'yellow',
            "plot_size": [14, 6],
            "image_size": (15, 15),
            "scatter_point_size": 50,
            "alpha": 0.5,
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
            "padding_factor": 3,
            "grid_size": (300, 300),
            "slice_positions": list(range(-25, 25, 10)),
            "rotation_matrix": [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        }
        self.interpolation: Dict[str, Any] = {
            "num_bins": 500,
            "smoothing_factor": 20.0,
            "curve_outlier_threshold": 10.0,
            "circle_point_margin": 10.0
        }
        self.ray_cast: Dict[str, Any] = {
            "strong_signal": 2,
            "medium_signal": 1,
            "no_signal": 0
        }

    def get(self, section: str, key: str) -> Any:
        return getattr(self, section).get(key)

# Initialize the config object
config = Config()
