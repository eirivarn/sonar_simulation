import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Union
from ideal_simulation.world_mapping import run_3d_mapping_simulation
import matplotlib.pyplot as plt

def format_filename(base_name: str, sonar_positions: List[Tuple[int, int]], angles: List[int]) -> str:
    pos_str = "_".join([f"s{i+1}_{y}_{x}" for i, (y, x) in enumerate(sonar_positions)])
    angle_str = "_".join([f"a{i+1}_{angle}" for i, angle in enumerate(angles)])
    os.makedirs('data', exist_ok=True)  # Ensure the 'data' directory exists
    return os.path.join('data', f"{base_name}_{pos_str}_{angle_str}.csv")

def calculate_x_distances_and_labels(x_circle, curve_x, num_labels=21):  # Using 21 to include -10 to 10
    labels = []
    max_distance = max(curve_x) - min(curve_x)
    step = max_distance / (num_labels - 1)
    left_boundary = x_circle - 10 * step
    right_boundary = x_circle + 10 * step

    for x in curve_x:
        if x < left_boundary:
            label = -10
        elif x > right_boundary:
            label = 10
        else:
            label = int(round((x - x_circle) / step))
        labels.append(label)

    return labels


def save_initial_results_to_csv(filename: str, data: List[Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, str, float, float, float, float]]]) -> None:
    rows = []
    for result in data:
        if result is None:
            continue
        # Unpack data
        x_circle, y_circle, radius, curve_x, curve_y, free_span_status, stability_percentage, enclosed_percentage, relative_distance, angle_degrees = result
        data_dict = {
            'x_circle': x_circle,
            'y_circle': y_circle,
            'radius': radius,
            'curve_x': list(curve_x),  # Convert numpy arrays to lists
            'curve_y': list(curve_y),  # Convert numpy arrays to lists
            'free_span_status': free_span_status,
            'stability_percentage': stability_percentage,
            'enclosed_percentage': enclosed_percentage,
            'relative_distance': relative_distance,
            'angle_degrees': angle_degrees
        }
        rows.append(data_dict)
    if rows:  # Only save if there is data
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    else:
        print(f"No data to save for {filename}")

def save_ground_truth_results_to_csv(filename: str, data: List[Union[None, Tuple[str, float, float, float, float]]]) -> None:
    rows = []
    for result in data:
        if result is None:
            continue
        # Unpack data
        free_span_status, stability_percentage, enclosed_percentage, relative_distance, angle_degrees = result
        data_dict = {
            'free_span_status': free_span_status,
            'stability_percentage': stability_percentage,
            'enclosed_percentage': enclosed_percentage,
            'relative_distance': relative_distance,
            'angle_degrees': angle_degrees
        }
        rows.append(data_dict)
    if rows:  # Only save if there is data
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    else:
        print(f"No data to save for {filename}")

def label_and_save_results(input_filename: str, output_filename: str):
    df = pd.read_csv(input_filename)
    labeled_rows = []
    for i, row in df.iterrows():
        x_circle = row['x_circle']
        curve_x = eval(row['curve_x'])
        curve_y = eval(row['curve_y'])
        labels = calculate_x_distances_and_labels(x_circle, curve_x)
        label_counts = {label: 0 for label in range(-10, 11)}
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
        signal_filename = format_filename('signal_results', sonar_positions_1, angles)
        save_initial_results_to_csv(signal_filename, signal_results)
        ground_truth_filename = 'data/ground_truth_results.csv'
        save_ground_truth_results_to_csv(ground_truth_filename, ground_truth_results)
        label_and_save_results(signal_filename, signal_filename.replace('.csv', '_with_labeling.csv'))
    else:
        print("No results to save.")
