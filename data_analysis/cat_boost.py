import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
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

# Function to train CatBoost model
def train_catboost_model(X_train, y_train):
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=100
    )
    train_pool = Pool(X_train, y_train)
    model.fit(train_pool)
    return model

# Main function to run the process
def main(signal_csv_paths, gt_csv_path):
    combined_df = pd.DataFrame()
    
    # Load, calculate differences, and combine data
    for signal_path in signal_csv_paths:
        signal_df, gt_df = load_data(signal_path, gt_csv_path)
        stability_diff = calculate_differences(signal_df, gt_df)
        signal_df['stability_diff'] = stability_diff
        combined_df = pd.concat([combined_df, signal_df], ignore_index=True)
    
    # Prepare data for modeling
    X, y = prepare_data_for_modeling(combined_df, combined_df['stability_diff'])
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Train the CatBoost model
    model = train_catboost_model(X_train, y_train)
    
    # Make predictions on the test set
    test_predictions = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f"Test RMSE: {test_rmse}")

    # Make predictions on the training set
    train_predictions = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    print(f"Train RMSE: {train_rmse}")
    
    # Print feature importances
    feature_importances = model.get_feature_importance()
    for i, col in enumerate(X_train.columns):
        print(f"Feature: {col}, Importance: {feature_importances[i]}")

# Specify the paths to your CSV files
signal_csv_paths = [
    'data/signal_results_s1_1000_2000_s2_1000_3740_a1_130_a2_230_with_labeling.csv',
    'data/signal_results_s1_1000_2000_a1_130_with_labeling.csv',
    'data/signal_results_s1_1000_3500_a1_220_with_labeling.csv',
    'data/signal_results_s1_1500_2870_a1_180_with_labeling.csv'
]
gt_csv_path = 'data/ground_truth_results.csv'

if __name__ == "__main__":
    main(signal_csv_paths, gt_csv_path)
