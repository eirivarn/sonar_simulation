import pandas as pd
import statsmodels.api as sm

def load_data(signal_path, gt_path):
    # Load the CSV files
    signal_df = pd.read_csv(signal_path)
    gt_df = pd.read_csv(gt_path)
    return signal_df, gt_df

def calculate_differences(signal_df, gt_df):
    # Calculate absolute differences in stability_percentage
    differences = abs(signal_df['stability_percentage'] - gt_df['stability_percentage'])
    return differences

def prepare_data_for_analysis(signal_df, differences):
    # Add the differences to the signal_df
    signal_df['stability_diff'] = differences
    
    # Prepare the feature matrix (X) and target vector (y)
    label_columns = ['label_' + str(i) for i in range(10)]  # Generate label column names
    X = signal_df[label_columns]
    y = signal_df['stability_diff']
    
    # Check for zero variance in the columns
    for col in label_columns:
        if X[col].var() == 0:
            print(f"Warning: Column {col} has zero variance.")
    
    if y.var() == 0:
        print("Warning: stability_diff has zero variance.")
    
    return X, y

def regression_analysis(X, y):
    # Add a constant term to the feature matrix (for intercept)
    X = sm.add_constant(X)
    
    # Fit the regression model
    model = sm.OLS(y, X).fit()
    
    # Print the summary of the regression analysis
    print(model.summary())
    
def main(signal_csv_path, gt_csv_path):
    signal_df, gt_df = load_data(signal_csv_path, gt_csv_path)
    stability_diff = calculate_differences(signal_df, gt_df)
    X, y = prepare_data_for_analysis(signal_df, stability_diff)
    regression_analysis(X, y)


# Specify the paths to your CSV files
signal_csv_path = 'data/signal_results_s1_1000_2000_a1_130_with_labeling.csv'
gt_csv_path = 'data/ground_truth_results.csv'

if __name__ == "__main__":
    main(signal_csv_path, gt_csv_path)
