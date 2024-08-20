import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function to load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Function to prepare data for modeling
def prepare_data_for_modeling(df):
    X = df.drop('abs_diff_stability', axis=1)
    y = df['abs_diff_stability']
    X.fillna(0, inplace=True)  # Filling missing values with 0, in case some labels are not present
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

def plot_feature_importances(importances, features):
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances by CatBoost Model")

    # Define a function to handle sorting feature names that may not always have an integer suffix
    def feature_sort_key(feature_name):
        parts = feature_name.split('_')
        # Attempt to convert the last part to an integer, if it fails, handle it gracefully
        try:
            return int(parts[-1])
        except ValueError:
            # Assign a default high value to ensure it sorts to the end if not numeric
            return float('inf')

    # Sorting the features by the extracted integer if present, or placing at the end if not
    sorted_features = sorted(zip(features, importances), key=lambda x: feature_sort_key(x[0]))
    features_sorted, importances_sorted = zip(*sorted_features)

    plt.bar(features_sorted, importances_sorted, color="b", align="center")
    plt.xticks(rotation=90)
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.show()
    
# Main function to run the process
def main(filepath):
    df = load_data(filepath)
    X, y = prepare_data_for_modeling(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = train_catboost_model(X_train, y_train)
    
    # Evaluate the model
    test_predictions = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f"Test RMSE: {test_rmse}")
    
    # Feature Importances
    feature_importances = model.get_feature_importance()
    plot_feature_importances(feature_importances, X.columns)

processed_data_path = 'data_processed/generated_processed_signal_data.csv'

if __name__ == "__main__":
    main(processed_data_path)
