import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Union
from ideal_simulation.world_mapping import run_3d_mapping_simulation
import matplotlib.pyplot as plt
from config import config
from matplotlib.colors import Normalize


def format_filename(base_name: str, sonar_positions: List[Tuple[int, int]], angles: List[int]) -> str:
    pos_str = "_".join([f"s{i+1}_{y}_{x}" for i, (y, x) in enumerate(sonar_positions)])
    angle_str = "_".join([f"a{i+1}_{angle}" for i, angle in enumerate(angles)])
    os.makedirs('data', exist_ok=True)
    return os.path.join('data', f"{base_name}_{pos_str}_{angle_str}.csv")

def calculate_x_distances_and_labels(x_circle, curve_x, radius):
    labels = []
    label_width = 50  # Each label spans 50 units (25 on each side)
    num_labels = 41  # Total labels from -20 to 20

    # Create the label starts centered around x_circle
    # Start from -20 label's start to 20 label's end
    start_label_center = x_circle - 20 * label_width
    label_starts = [start_label_center + i * label_width for i in range(num_labels)]
    label_ranges = [(label_starts[i] - label_width / 2, label_starts[i] + label_width / 2) for i in range(len(label_starts))]

    # Debug: print label ranges
    # print(f"Label -21: Range from -inf to {label_ranges[0][0]}")
    # for i, range_ in enumerate(label_ranges):
    #     print(f"Label {i - 20}: Range from {range_[0]} to {range_[1]}")
    # print(f"Label 21: Range from {label_ranges[-1][1]} to inf")

    for x in curve_x:
        if x < label_ranges[0][0]:
            label = -21  # Anything further left than the start of the first label
        elif x > label_ranges[-1][1]:
            label = 21   # Anything further right than the end of the last label
        else:
            # Assign label based on position
            for idx, (start, end) in enumerate(label_ranges):
                if start <= x < end:
                    label = idx - 20
                    break
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

        print(f"Processing row {i}: x_circle = {x_circle}, radius = {radius}")
        labels = calculate_x_distances_and_labels(x_circle, curve_x, radius)
        
        label_counts = {label: 0 for label in range(-21, 22)}  # From -21 to 21
        for label in labels:
            label_counts[label] += 1

        row_data = row.to_dict()
        row_data.update({f'label_{label}': count for label, count in label_counts.items()})
        labeled_rows.append(row_data)
        
        if config.show_plots:
            fig, ax = plt.subplots()
            cmap = plt.get_cmap('coolwarm', 43)  # Adjusted for new label range
            norm = plt.Normalize(vmin=-21, vmax=21)
            scatter = ax.scatter(curve_x, curve_y, c=labels, cmap=cmap, norm=norm, alpha=0.5)
            plt.colorbar(scatter, label='Label')
            ax.set_title(f'Curve Points Colored by Label for row {i}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.grid(True)
            
            # Add text labels to the plot
            for j, txt in enumerate(labels):
                ax.annotate(txt, (curve_x[j], curve_y[j]), fontsize=8, alpha=0.7)

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
            signal_filename = format_filename('generated_signal_results.csv', sonar_positions_1, angles)
            ground_truth_filename = 'data/generated_ground_truth_results.csv'
        
        save_initial_results_to_csv(signal_filename, signal_results)
        save_ground_truth_results_to_csv(ground_truth_filename, ground_truth_results)
        label_and_save_results(signal_filename, signal_filename.replace('.csv', '_with_labeling.csv'))
    else:
        print("No results to save.")
