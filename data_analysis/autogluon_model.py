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

# Function to prepare data for modeling
def prepare_data_for_modeling(df):
    label_columns = [f'label_{i}' for i in range(-21, 22)]
    X = df[label_columns]
    y = df['abs_diff_stability']
    return X, y

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

def plot_predictions(ground_truth, predictions, title):
    plt.figure(figsize=(12, 6))
    plt.plot(ground_truth.index, ground_truth, label='Ground Truth', color='blue')
    plt.plot(ground_truth.index, predictions, label='Predictions', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_new_data(model_path, new_data_path):
    # Load the new data
    new_df = load_data(new_data_path)
    
    # Prepare data for modeling
    X_new, y_new = prepare_data_for_modeling(new_df)
    
    # Load the trained predictor
    predictor = TabularPredictor.load(model_path)
    
    # Create the dataset for AutoGluon
    new_data = pd.concat([X_new, y_new], axis=1)
    new_data.columns = [f'label_{i}' for i in range(-21, 22)] + ['abs_diff_stability']
    new_data = TabularDataset(new_data)
    
    # Make predictions
    new_predictions = predictor.predict(new_data)
    
    # Calculate evaluation metrics
    new_rmse = np.sqrt(mean_squared_error(y_new, new_predictions))
    print(f"New Data RMSE: {new_rmse}")
    
    # Plot predictions vs ground truth
    plot_predictions(y_new, new_predictions, 'Comparison of Predictions and Ground Truth on New Data')
    
# Main function to run the process
def main(filepath, random_split=True, model_path=None, new_data_path=None):
    if model_path and new_data_path:
        # Evaluate on new dataset using an existing model
        evaluate_new_data(model_path, new_data_path)
    else:
        df = load_data(filepath)
        
        # Prepare data
        X, y = prepare_data_for_modeling(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) if random_split else (X[:int(len(X) * 0.9)], X[int(len(X) * 0.9):], y[:int(len(y) * 0.9)], y[int(len(y) * 0.9):])
        
        # Create training and test datasets for AutoGluon
        train_data = pd.concat([X_train, y_train], axis=1)
        train_data.columns = [f'label_{i}' for i in range(-21, 22)] + ['abs_diff_stability']
        train_data = TabularDataset(train_data)
        
        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.columns = [f'label_{i}' for i in range(-21, 22)] + ['abs_diff_stability']
        test_data = TabularDataset(test_data)

        # Train the model
        predictor = train_with_autogluon(train_data)
        
        print("AutoGluon model training complete.")
        
        # Predictions and evaluation for test data
        test_predictions = predictor.predict(test_data)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        print(f"Test RMSE: {rmse}")
        plot_predictions(y_test, test_predictions, 'Comparison of Predictions and Ground Truth on Test Data')


# Specify the path to your processed CSV file
processed_data_path = 'data_processed/generated_processed_signal_data.csv'
# Specify the path to the model directory
model_directory = 'AutogluonModels/ag-20240802_052514'  # Path where the trained model is saved
# Specify the path to the new data
new_data_path = 'data_processed/loaded_processed_signal_data.csv'

if __name__ == "__main__":
       main(new_data_path, random_split=False)

