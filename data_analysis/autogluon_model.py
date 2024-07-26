import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to load data
def load_data(signal_path, gt_path):
    signal_df = pd.read_csv(signal_path)
    gt_df = pd.read_csv(gt_path)
    return signal_df, gt_df

# Function to calculate differences
def calculate_differences(signal_df, gt_df):
    differences = abs(signal_df['stability_percentage'] - gt_df['stability_percentage'])
    return differences

# Function to prepare data for modeling
def prepare_data_for_modeling(signal_df, differences):
    signal_df['stability_diff'] = differences
    label_columns = [f'label_{i}' for i in range(10)]
    X = signal_df[label_columns]
    y = signal_df['stability_diff']
    return X, y

# Function to remove outliers
def remove_outliers(X, y, z_thresh=3):
    from scipy.stats import zscore
    z_scores_X = np.abs(zscore(X))
    z_scores_y = np.abs(zscore(y))
    
    filter_X = (z_scores_X < z_thresh).all(axis=1)
    filter_y = z_scores_y < z_thresh
    
    combined_filter = filter_X & filter_y
    
    X_clean = X[combined_filter]
    y_clean = y[combined_filter]
    
    return X_clean, y_clean

# Function to train model with AutoGluon
def train_with_autogluon(train_data):
    predictor = TabularPredictor(label='stability_diff').fit(
        train_data,
        hyperparameters={
            'CAT': {},
            'XGB': {},
            'RF': [{'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
        }
    )
    return predictor

# Main function to run the process
def main(signal_csv_path, gt_csv_path):
    # Load and preprocess data
    signal_df, gt_df = load_data(signal_csv_path, gt_csv_path)
    stability_diff = calculate_differences(signal_df, gt_df)
    X, y = prepare_data_for_modeling(signal_df, stability_diff)
    X_clean, y_clean = remove_outliers(X, y)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    
    # Train the model
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.columns = [f'label_{i}' for i in range(10)] + ['stability_diff']
    train_data = TabularDataset(train_data)
    predictor = train_with_autogluon(train_data)
    
    print("AutoGluon model training complete.")
    
    # Evaluate on the test set
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.columns = [f'label_{i}' for i in range(10)] + ['stability_diff']
    test_data = TabularDataset(test_data)

    # Make predictions on the test set
    test_predictions = predictor.predict(test_data)
    test_true = y_test

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(test_true, test_predictions))
    print(f"Test RMSE: {rmse}")

    # Evaluate on the training set
    train_predictions = predictor.predict(train_data)
    train_true = y_train

    # Calculate evaluation metrics
    train_rmse = np.sqrt(mean_squared_error(train_true, train_predictions))
    print(f"Train RMSE: {train_rmse}")

    # Print leaderboard
    leaderboard = predictor.leaderboard()
    print(leaderboard)

# Specify the paths to your CSV files
signal_csv_path = 'data/signal_results_s1_1000_2000_a1_130_with_labeling.csv'
gt_csv_path = 'data/ground_truth_results.csv'

if __name__ == "__main__":
    main(signal_csv_path, gt_csv_path)
