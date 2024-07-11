import pandas as pd
from typing import Tuple, List, Union
import numpy as np
import os 
import math


def format_filename(base_name: str, sonar_positions: List[Tuple[int, int]], angles: List[int]) -> str:
    pos_str = "_".join([f"s{i+1}_{y}_{x}" for i, (y, x) in enumerate(sonar_positions)])
    angle_str = "_".join([f"a{i+1}_{angle}" for i, angle in enumerate(angles)])
    os.makedirs('data', exist_ok=True)  # Ensure the 'data' directory exists
    return os.path.join('data', f"{base_name}_{pos_str}_{angle_str}.csv")

def save_results_to_csv(filename: str, data_type: str, data: List[Union[None, Tuple[float, float, float, np.ndarray, np.ndarray, str, float, float, float, float]]]) -> None:
    rows = []
    for result in data:
        if result is None:
            continue
        # Unpack data
        x_circle, y_circle, radius, curve_x, curve_y, free_span_status, stability_percentage, enclosed_percentage, relative_distance, angle_degrees = result
        
        data = {
            'free_span_status': free_span_status,
            'stability_percentage': stability_percentage,
            'enclosed_percentage': enclosed_percentage,
            'relative_distance': relative_distance,
            'angle_degrees': angle_degrees,
        }
        rows.append(data)
        
    if rows:  # Only save if there is data
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    else:
        print(f"No data to save for {filename}")
        
def compare_files(ground_truth_file_path, test_file_path):
    # Load the CSV files, potentially resetting the index if the first column isn't meant to be used as an index
    ground_truth_df = pd.read_csv(ground_truth_file_path)
    test_df = pd.read_csv(test_file_path)
    
    # Reset index if the first column is not an identifier
    ground_truth_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Ensure both DataFrames have the same structure
    if test_df.columns.tolist() != ground_truth_df.columns.tolist():
        raise ValueError("Columns do not match between test and ground truth files.")

    # Initialize dictionaries to store the results
    difference_sums_dict = {}
    average_errors_dict = {}
    largest_differences_dict = {}
    largest_difference_indices_dict = {}
    
    # Handle the first column (usually an identifier or a non-numeric column)
    first_column_name = test_df.columns[0]
    difference_sums_dict[first_column_name] = (test_df[first_column_name] != ground_truth_df[first_column_name]).sum()
    average_errors_dict[first_column_name] = None  # Average error doesn't apply to non-numeric comparison
    largest_differences_dict[first_column_name] = None  # Largest difference doesn't apply to non-numeric comparison
    largest_difference_indices_dict[first_column_name] = None  # Row index doesn't apply
    
    # Calculate the absolute differences for the numeric columns
    for column in test_df.columns[1:]:
        absolute_differences = (test_df[column] - ground_truth_df[column]).abs()
        difference_sums_dict[column] = absolute_differences.sum()
        average_errors_dict[column] = absolute_differences.mean()
        largest_differences_dict[column] = absolute_differences.max()
        largest_difference_indices_dict[column] = absolute_differences.idxmax()

    print("Difference counts, sum of absolute differences, average errors, largest differences, and largest difference row indices by column:")
    for column in difference_sums_dict.keys():
        print(f"{column}:")
        print(f"  Sum of differences: {difference_sums_dict[column]}")
        if average_errors_dict[column] is not None:
            print(f"  Average error: {average_errors_dict[column]}")
        else:
            print(f"  Average error: N/A (non-numeric comparison)")
        if largest_differences_dict[column] is not None:
            print(f"  Largest difference: {largest_differences_dict[column]}")
            print(f"  Row index of largest difference: {largest_difference_indices_dict[column]}")
        else:
            print(f"  Largest difference: N/A (non-numeric comparison)")
            print(f"  Row index of largest difference: N/A")
    
    return difference_sums_dict, average_errors_dict, largest_differences_dict, largest_difference_indices_dict