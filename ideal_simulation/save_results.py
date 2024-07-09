import pandas as pd
from typing import Tuple, List, Union
import numpy as np
import os 

def format_filename(base_name: str, sonar_positions: List[Tuple[int, int]], angles: List[int]) -> str:
    pos_str = "_".join([f"s{i+1}_{y}_{x}" for i, (y, x) in enumerate(sonar_positions)])
    angle_str = "_".join([f"a{i+1}_{angle}" for i, angle in enumerate(angles)])
    # Ensure the 'data' directory exists
    os.makedirs('data', exist_ok=True)
    return os.path.join('data', f"{base_name}_{pos_str}_{angle_str}.csv")

def save_results_to_csv(filename: str, data: List[Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, float, float]]]) -> None:
    rows = []
    for result in data:
        if result is None:
            continue
        x_circle, y_circle, radius, curve_x, curve_y, free_span_status, stability_percentage = result[:7]
        ground_truth = result[7] if len(result) > 7 else None
        
        signal_data = {
            'free_span_status': free_span_status,
            'stability_percentage': stability_percentage,
        }
        rows.append(signal_data)
        
        if ground_truth is not None:
            gt_x_circle, gt_y_circle, gt_radius, gt_curve_x, gt_curve_y, gt_free_span_status, gt_stability_percentage = ground_truth
            ground_truth_data = {
                'gt_free_span_status': gt_free_span_status,
                'gt_stability_percentage': gt_stability_percentage,
            }
            rows.append(ground_truth_data)

    if rows:  # Only save if there is data
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    else:
        print(f"No data to save for {filename}")