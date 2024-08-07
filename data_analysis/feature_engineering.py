import pandas as pd
import os

def load_data(signal_path, gt_path):
    signal_df = pd.read_csv(signal_path)
    gt_df = pd.read_csv(gt_path)
    return signal_df, gt_df

def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

def add_features(signal_df, gt_df):
    # Remove outliers in 'stability_percentage' before calculating differences
    signal_df = remove_outliers(signal_df, 'stability_percentage').copy()
    gt_df = remove_outliers(gt_df, 'stability_percentage').copy()

    # Calculate the absolute difference in stability percentage
    stability_diff = abs(signal_df['stability_percentage'] - gt_df['stability_percentage'])
    signal_df.loc[:, 'abs_diff_stability'] = stability_diff  # Using .loc to avoid SettingWithCopyWarning
    
    # Include label columns from signal data
    label_columns = [col for col in signal_df.columns if 'label_' in col]
    signal_df = pd.concat([signal_df[label_columns], signal_df[['abs_diff_stability']]], axis=1).copy()

    return signal_df

def save_processed_data(signal_df, folder='data_processed'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, 'processed_signal_data.csv')
    signal_df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

def main(signal_csv_paths, gt_csv_path):
    combined_df = pd.DataFrame()
    for signal_path in signal_csv_paths:
        signal_df, gt_df = load_data(signal_path, gt_csv_path)
        signal_df = add_features(signal_df, gt_df)
        combined_df = pd.concat([combined_df, signal_df], ignore_index=True)
    
    save_processed_data(combined_df)

# Specify the paths to your CSV files
signal_csv_paths = [
    'data/pos_&_neg_values/signal_results_s1_1000_2000_s2_1000_3740_a1_130_a2_230_with_labeling.csv',
    'data/pos_&_neg_values/signal_results_s1_1000_2000_a1_130_with_labeling.csv',
    'data/pos_&_neg_values/signal_results_s1_1500_2870_a1_180_with_labeling.csv',
    'data/pos_&_neg_values/signal_results_s1_1000_3500_a1_220_with_labeling.csv',
    'data/pos_&_neg_values/signal_results_s1_1000_2000_a1_150_with_labeling.csv',
    'data/pos_&_neg_values/signal_results_s1_1000_3500_a1_200_with_labeling.csv'
]

gt_csv_path = 'data/ground_truth_results.csv'

if __name__ == "__main__":
    main(signal_csv_paths, gt_csv_path)
