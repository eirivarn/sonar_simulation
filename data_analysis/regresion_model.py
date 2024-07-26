import pandas as pd
import statsmodels.api as sm

def load_data(signal_path, gt_path):
    signal_df = pd.read_csv(signal_path)
    gt_df = pd.read_csv(gt_path)
    return signal_df, gt_df

def calculate_differences(signal_df, gt_df):
    differences = abs(signal_df['stability_percentage'] - gt_df['stability_percentage'])
    return differences

def prepare_data_for_analysis(signal_df, differences):
    signal_df['stability_diff'] = differences
    
    label_columns = [col for col in signal_df.columns if col.startswith('label_')]
    X = signal_df[label_columns]
    y = signal_df['stability_diff']
    
    for col in label_columns:
        if X[col].var() == 0:
            print(f"Warning: Column {col} has zero variance.")
    
    if y.var() == 0:
        print("Warning: stability_diff has zero variance.")
    
    return X, y

def regression_analysis(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

def main(signal_csv_paths, gt_csv_path):
    combined_df = pd.DataFrame()
    for signal_path in signal_csv_paths:
        signal_df, gt_df = load_data(signal_path, gt_csv_path)
        stability_diff = calculate_differences(signal_df, gt_df)
        signal_df['stability_diff'] = stability_diff
        combined_df = pd.concat([combined_df, signal_df], ignore_index=True)
    
    X, y = prepare_data_for_analysis(combined_df, combined_df['stability_diff'])
    regression_analysis(X, y)

signal_csv_paths = [
    'data/signal_results_s1_1000_2000_s2_1000_3740_a1_130_a2_230_with_labeling.csv',
    'data/signal_results_s1_1000_2000_a1_130_with_labeling.csv',
    'data/signal_results_s1_1000_3500_a1_220_with_labeling.csv',
    'data/signal_results_s1_1500_2870_a1_180_with_labeling.csv'
]
gt_csv_path = 'data/ground_truth_results.csv'

if __name__ == "__main__":
    main(signal_csv_paths, gt_csv_path)
