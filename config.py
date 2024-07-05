class Config:
    def __init__(self):
        self.dimensions = (500, 1000) #(y, x)
        self.pipe_center = (90, 500)   #(y, x)
        self.pipe_radius = 30   
        self.combined_mesh_path = ['/Users/eirikvarnes/code/blender/combined.obj']
        self.seperate_mesh_paths = ['/Users/eirikvarnes/code/blender/seafloor.obj', '/Users/eirikvarnes/code/blender/pipeline.obj']
        
        self.sonar = {
            "max_range": 500,
            "angle_width": 60,
            "num_rays": 120
        }
        self.clustering = {
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
        self.plotting = {
            "room_color": 'gray',
            "sonar_position_color": 'red',
            "sonar_ray_color": 'yellow',
            "plot_size": [14, 6]
        }

    def get(self, section: str, key: str):
        return getattr(self, section).get(key)

# Initialize the config object
config = Config()
