import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Function to prepare data for modeling
def prepare_data_for_modeling(df):
    label_columns = [f'label_{i}' for i in range(10)]
    X = df[label_columns]
    y = df['abs_diff_stability']
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
    predictor = TabularPredictor(label='abs_diff_stability').fit(
        train_data,
        hyperparameters={
            'CAT': {},
            'XGB': {},
            'RF': [{'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
        }
    )
    return predictor

# Main function to run the process
def main(filepath, random_split=True):
    df = load_data(filepath)
    
    # Prepare and clean data
    X, y = prepare_data_for_modeling(df)
    X_clean, y_clean = remove_outliers(X, y)
    
    # Split the data into training and test sets based on user choice
    if random_split:
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.9, random_state=42)
    else:
        split_index = int(len(X_clean) * 0.9)
        X_train, X_test = X_clean[:split_index], X_clean[split_index:]
        y_train, y_test = y_clean[:split_index], y_clean[split_index:]
    
    # Create training and test datasets for AutoGluon
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.columns = [f'label_{i}' for i in range(10)] + ['abs_diff_stability']
    train_data = TabularDataset(train_data)
    
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.columns = [f'label_{i}' for i in range(10)] + ['abs_diff_stability']
    test_data = TabularDataset(test_data)

    # Train the model
    predictor = train_with_autogluon(train_data)
    
    print("AutoGluon model training complete.")
    
    # Make predictions on the test set
    test_predictions = predictor.predict(test_data)
    test_true = y_test

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(test_true, test_predictions))
    print(f"Test RMSE: {rmse}")

    # Make predictions on the training set
    train_predictions = predictor.predict(train_data)
    train_true = y_train

    # Calculate evaluation metrics
    train_rmse = np.sqrt(mean_squared_error(train_true, train_predictions))
    print(f"Train RMSE: {train_rmse}")

    # Print leaderboard
    leaderboard = predictor.leaderboard()
    print(leaderboard)

# Specify the path to your processed CSV file
processed_data_path = 'data_processed/processed_signal_data.csv'

if __name__ == "__main__":
    main(processed_data_path, random_split=False)
