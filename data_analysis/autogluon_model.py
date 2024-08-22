import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function to load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    if df.empty:
        print("Error: Data file is empty.")
    else:
        print(f"Data loaded successfully with {df.shape[0]} records.")
    return df

# Function to load signal and GT data separately
def load_signal_and_gt(signal_path, gt_path):
    signal_df = pd.read_csv(signal_path)
    gt_df = pd.read_csv(gt_path)
    return signal_df, gt_df

# Function to remove outliers
def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

# Function to add features
def add_features(signal_df, gt_df):
    signal_df = remove_outliers(signal_df, 'stability_percentage').copy()
    gt_df = remove_outliers(gt_df, 'stability_percentage').copy()
    stability_diff = signal_df['stability_percentage'] - gt_df['stability_percentage']
    signal_df.loc[:, 'diff_stability'] = stability_diff
    label_columns = [col for col in signal_df.columns if 'label_' in col]
    signal_df = pd.concat([signal_df[label_columns], signal_df[['stability_percentage', 'diff_stability']]], axis=1).copy()
    return signal_df, gt_df

# Function to prepare data for modeling
def prepare_data_for_modeling(df):
    label_columns = [f'label_{i}' for i in range(-21, 22)]
    X = df[label_columns]
    y = df['diff_stability']
    return X, y

# Function to calculate baseline RMSE using mean of the target
def calculate_baseline_rmse(y_train, y_test):
    mean_prediction = np.mean(y_train)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, [mean_prediction] * len(y_test)))
    return baseline_rmse, mean_prediction

# Function to train model with AutoGluon
def train_with_autogluon(train_data):
    predictor = TabularPredictor(label='diff_stability').fit(
        train_data,
        hyperparameters={
            'CAT': {},
            'XGB': {},
            'RF': [{'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
        }
    )
    return predictor

# Function to calculate prediction intervals
def calculate_prediction_intervals(predictions, errors, confidence_level=0.95):
    std_dev = np.std(errors)
    z_score = 1.96  # for 95% confidence
    margin_of_error = z_score * std_dev
    lower_bounds = predictions - margin_of_error
    upper_bounds = predictions + margin_of_error
    return lower_bounds, upper_bounds

def plot_predictions_with_intervals(ground_truth, predictions, lower_bounds, upper_bounds, title):
    plt.figure(figsize=(12, 6))
    plt.plot(ground_truth.index, ground_truth, label='Ground Truth', color='blue')
    plt.plot(ground_truth.index, predictions, label='Predictions', color='red', linestyle='--')
    plt.fill_between(ground_truth.index, lower_bounds, upper_bounds, color='gray', alpha=0.2, label='Prediction Interval')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparison(baseline_rmse, model_rmse):
    labels = ['Baseline RMSE', 'Model RMSE']
    values = [baseline_rmse, model_rmse]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['gray', 'green'])
    plt.title('Comparison of Baseline RMSE vs. Model RMSE')
    plt.ylabel('RMSE')
    plt.ylim(0, max(values) * 1.1)
    plt.show()

def plot_test_error_comparison(signal_df, gt_df, corrected_signal_df, split_index):
    # Focus only on the test data
    signal_test_df = signal_df.iloc[split_index:].reset_index(drop=True)
    gt_test_df = gt_df.iloc[split_index:].reset_index(drop=True)
    corrected_test_df = corrected_signal_df.iloc[split_index:].reset_index(drop=True)

    # Calculate errors for the test dataset
    original_error = np.abs(signal_test_df['stability_percentage'] - gt_test_df['stability_percentage'])
    corrected_error = np.abs(corrected_test_df['stability_percentage_corrected'] - gt_test_df['stability_percentage'])
    
    # Calculate the mean absolute errors (MAE) for the test data
    original_mae = np.mean(original_error)
    corrected_mae = np.mean(corrected_error)
    
    # Print the MAE for both original and corrected errors
    print(f"Test Data - Mean Absolute Error (Original): {original_mae:.4f}")
    print(f"Test Data - Mean Absolute Error (Corrected): {corrected_mae:.4f}")
    
    # Plotting the errors for the test data
    plt.figure(figsize=(12, 6))
    plt.plot(original_error.index, original_error, label='Original Error (Test)', color='blue', linestyle='--')
    plt.plot(corrected_error.index, corrected_error, label='Corrected Error (Test)', color='red')
    plt.title('Test Data: Error Comparison (Original vs Corrected)')
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Error')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def main(filepath, signal_csv_path=None, gt_csv_path=None, model_path=None):
    df = load_data(filepath)
    
    # Prepare data
    X, y = prepare_data_for_modeling(df)
    
    # Fixed split of data: 75% training, 25% testing
    split_index = int(len(X) * 0.75)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Calculate baseline RMSE and prediction intervals
    baseline_rmse, mean_prediction = calculate_baseline_rmse(y_train, y_test)
    print(f"Baseline RMSE: {baseline_rmse}")
    
    baseline_predictions = np.full_like(y_test, mean_prediction)
    baseline_errors = baseline_predictions - y_test
    baseline_lower_bounds, baseline_upper_bounds = calculate_prediction_intervals(baseline_predictions, baseline_errors)

    # Create training and test datasets for AutoGluon
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.columns = [f'label_{i}' for i in range(-21, 22)] + ['diff_stability']
    train_data = TabularDataset(train_data)
    
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.columns = [f'label_{i}' for i in range(-21, 22)] + ['diff_stability']
    test_data = TabularDataset(test_data)

    # Train the model
    predictor = train_with_autogluon(train_data)
    print("AutoGluon model training complete.")
    
    # Predictions and evaluation for test data
    model_predictions = predictor.predict(test_data)
    model_errors = model_predictions - y_test
    model_rmse = np.sqrt(mean_squared_error(y_test, model_predictions))
    print(f"Model RMSE on Test Data: {model_rmse}")
    
    # Calculate prediction intervals for the model
    model_lower_bounds, model_upper_bounds = calculate_prediction_intervals(model_predictions, model_errors)
    
    # Find the largest `stability_diff` value in the test set
    max_stability_diff = y_test.max()
    print(f"Largest stability_diff value in the test set: {max_stability_diff}")
    
    # Find the largest error between the model's prediction and the actual value
    max_error = np.abs(model_errors).max()
    print(f"Largest error between prediction and actual value on Test Data: {max_error}")
    
    # Plot comparison of baseline RMSE vs. model RMSE
    plot_comparison(baseline_rmse, model_rmse)
    
    # Plot predictions vs ground truth for baseline model with prediction intervals
    plot_predictions_with_intervals(y_test, baseline_predictions, baseline_lower_bounds, baseline_upper_bounds, 'Baseline Predictions with Prediction Intervals')
    
    # Plot predictions vs ground truth for the trained model with prediction intervals
    plot_predictions_with_intervals(y_test, model_predictions, model_lower_bounds, model_upper_bounds, 'Model Predictions with Prediction Intervals')

    if signal_csv_path and gt_csv_path and model_path:
        # Load and process the original signal and GT data
        signal_df, gt_df = load_signal_and_gt(signal_csv_path, gt_csv_path)
        signal_df, gt_df = add_features(signal_df, gt_df)

        # Prepare the data for prediction
        train_data_for_correction = TabularDataset(signal_df.drop(columns=['diff_stability', 'stability_percentage']))
        
        # Predict the stability difference using the trained model
        predicted_diff = predictor.predict(train_data_for_correction)
        
        # Correct the signal stability using the model's prediction
        corrected_signal_df = signal_df.copy()
        corrected_signal_df['stability_percentage_corrected'] = corrected_signal_df['stability_percentage'] - predicted_diff
        
        # Simplified plot comparing the original and corrected errors on test data only
        plot_test_error_comparison(signal_df, gt_df, corrected_signal_df, split_index)

# Specify the paths to your processed CSV file, signal CSV, GT CSV, and model
processed_data_path = 'data_processed/processed_signal_data.csv'
signal_csv_path = 'data/pos_&_neg_values/s1_3500_2665_s2_3500_2965_a1_175_a2_185/signal_results_s1_3500_2665_s2_3500_2965_a1_175_a2_185_with_labeling.csv'
gt_csv_path = 'data/pos_&_neg_values/ground_truth_results.csv'
model_path = 'path/to/your/saved/model'

if __name__ == "__main__":
    main(processed_data_path, signal_csv_path, gt_csv_path, model_path)