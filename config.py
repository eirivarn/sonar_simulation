from typing import Tuple, Dict, Any, List

class Config:
    def __init__(self):
        self.dimensions: Tuple[int, int] = (500, 1000)  # (y, x)
        self.pipe_center: Tuple[int, int] = (90, 500)   # (y, x)
        self.pipe_radius: int = 30
        self.combined_mesh_path: List[str] = ['/Users/eirikvarnes/code/blender/combined.obj']
        self.seperate_mesh_paths: List[str] = ['/Users/eirikvarnes/code/blender/seafloor.obj', '/Users/eirikvarnes/code/blender/pipeline.obj']
        
        self.sonar: Dict[str, Any] = {
            "max_range": 500,
            "angle_width": 60,
            "num_rays": 120
        }
        self.clustering: Dict[str, Dict[str, Any]] = {
            "dbscan": {
                "eps": 0.5,
                "min_samples": 10
            },
            "kmeans": {
                "n_clusters": 3
            },
            "agglomerative": {
                "n_clusters": 3
            },
            "ransac": {
                "min_samples": 10,
                "residual_threshold": 9,
                "max_trials": 1000
            }
        }
        self.plotting: Dict[str, Any] = {
            "room_color": 'gray',
            "sonar_position_color": 'red',
            "sonar_ray_color": 'yellow',
            "plot_size": [14, 6]
        }
        self.ground_wave: Dict[str, Any] = {
            "amplitude": 2,
            "frequency": 0.05,
            "base_level": 45,
            "repeat": 1024
        }
        self.ray_cast: Dict[str, float] = {
            "strong_signal": 1.5,
            "medium_signal": 0.5
        }

    def get(self, section: str, key: str) -> Any:
        return getattr(self, section).get(key)

# Initialize the config object
config = Config()
