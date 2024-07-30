import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Union
from ideal_simulation.world_mapping import run_3d_mapping_simulation
import matplotlib.pyplot as plt
from config import config

def format_filename(base_name: str, sonar_positions: List[Tuple[int, int]], angles: List[int]) -> str:
    pos_str = "_".join([f"s{i+1}_{y}_{x}" for i, (y, x) in enumerate(sonar_positions)])
    angle_str = "_".join([f"a{i+1}_{angle}" for i, angle in enumerate(angles)])
    os.makedirs('data', exist_ok=True)
    return os.path.join('data', f"{base_name}_{pos_str}_{angle_str}.csv")

def calculate_x_distances_and_labels(x_circle, curve_x, radius):
    labels = []
    label_range = np.linspace(x_circle - 10 * radius, x_circle + 10 * radius, num=21)  # 21 points for labels from -10 to 10
    
    for x in curve_x:
        if x < x_circle - 10 * radius:
            label = -11  # More than 10 units to the left
        elif x > x_circle + 10 * radius:
            label = 11  # More than 10 units to the right
        else:
            # Calculate the theoretical label index
            label_index = np.searchsorted(label_range, x, side='right') - 1
            label = label_index - 10  # Convert index to range from -10 to 10
        labels.append(label)
    return labels

def save_initial_results_to_csv(filename: str, data: List[Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, str, float, float, float, float]]]) -> None:
    rows = []
    for result in data:
        if result is None:
            continue
        x_circle, y_circle, radius, curve_x, curve_y, free_span_status, stability_percentage, enclosed_percentage, relative_distance, angle_degrees = result
        data_dict = {
            'x_circle': x_circle,
            'y_circle': y_circle,
            'radius': radius,
            'curve_x': list(curve_x),
            'curve_y': list(curve_y),
            'free_span_status': free_span_status,
            'stability_percentage': stability_percentage,
            'enclosed_percentage': enclosed_percentage,
            'relative_distance': relative_distance,
            'angle_degrees': angle_degrees
        }
        rows.append(data_dict)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    else:
        print(f"No data to save for {filename}")

def save_ground_truth_results_to_csv(filename: str, data: List[Union[None, Tuple[str, float, float, float, float]]]) -> None:
    rows = []
    for result in data:
        if result is None:
            continue
        x_circle_translated, y_circle_translated, radius, curve_x_translated, curve_y_translated, free_span_status, stability_percentage, enclosed_percentage, relative_distance, angle_degrees = result
        data_dict = {
            'free_span_status': free_span_status,
            'stability_percentage': stability_percentage,
            'enclosed_percentage': enclosed_percentage,
            'relative_distance': relative_distance,
            'angle_degrees': angle_degrees
        }
        rows.append(data_dict)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    else:
        print(f"No data to save for {filename}")

def label_and_save_results(input_filename: str, output_filename: str):
    df = pd.read_csv(input_filename)
    labeled_rows = []
    for i, row in df.iterrows():
        x_circle = row['x_circle']
        radius = row['radius']
        curve_x = eval(row['curve_x'])
        curve_y = eval(row['curve_y'])
        labels = calculate_x_distances_and_labels(x_circle, curve_x, radius)
        # Initialize the dictionary to include all possible labels and extra labels for distant points
        label_counts = {label: 0 for label in range(-11, 12)}  # From -11 to 11
        for label in labels:
            label_counts[label] += 1
        row_data = row.to_dict()
        row_data.pop('x_circle')
        row_data.pop('y_circle')
        row_data.pop('radius')
        row_data.pop('curve_x')
        row_data.pop('curve_y')
        row_data.update({f'label_{label}': count for label, count in label_counts.items()})
        labeled_rows.append(row_data)
        print(f"Plotting row {i}")
        plt.scatter(curve_x, curve_y, c=labels, cmap='viridis')
        plt.colorbar(label='Label')
        plt.title(f'Curve Points Colored by Label for row {i}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    if labeled_rows:
        labeled_df = pd.DataFrame(labeled_rows)
        labeled_df.to_csv(output_filename, index=False)
    else:
        print(f"No data to save for {output_filename}")


def run_detection_evaluation(sonar_positions_1, angles, slice_positions):
    results = run_3d_mapping_simulation(sonar_positions_1, angles, slice_positions)
    signal_results = [r[:10] for r in results if r is not None and len(r) >= 10]
    ground_truth_results = [r[10] for r in results if r is not None and len(r) > 10]
    if results:
        if config.load_data:
            signal_filename = format_filename('signal_results', sonar_positions_1, angles)
            ground_truth_filename = 'data/ground_truth_results.csv'
        else: 
            signal_filename = 'data/generated_signal_results.csv'
            ground_truth_filename = 'data/generated_ground_truth_results.csv'
        
        save_initial_results_to_csv(signal_filename, signal_results)
        save_ground_truth_results_to_csv(ground_truth_filename, ground_truth_results)
        label_and_save_results(signal_filename, signal_filename.replace('.csv', '_with_labeling.csv'))
        
        
    else:
        print("No results to save.")
