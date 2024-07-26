import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    return X, y

def plot_target_distribution(y):
    sns.histplot(y, kde=True)
    plt.title("Target Variable Distribution")
    plt.xlabel("Stability Difference")
    plt.ylabel("Frequency")
    plt.show()

def random_forest_analysis(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best parameters found by Grid Search: {best_params}")
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    feature_importances = best_rf.feature_importances_
    for i, col in enumerate(X.columns):
        print(f"Feature: {col}, Importance: {feature_importances[i]}")
    plt.barh(X.columns, feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importances')
    plt.show()

def main(signal_csv_paths, gt_csv_path):
    combined_df = pd.DataFrame()
    for signal_path in signal_csv_paths:
        signal_df, gt_df = load_data(signal_path, gt_csv_path)
        stability_diff = calculate_differences(signal_df, gt_df)
        
        # Diagnostic: Print some summary statistics of the raw differences
        print(f"Summary statistics for {signal_path}:")
        print(stability_diff.describe())
        print(f"Mean difference: {stability_diff.mean()}")
        print(f"Median difference: {stability_diff.median()}")
        
        signal_df['stability_diff'] = stability_diff
        combined_df = pd.concat([combined_df, signal_df], ignore_index=True)
    
    plot_target_distribution(combined_df['stability_diff'])
    X, y = prepare_data_for_analysis(combined_df, combined_df['stability_diff'])
    random_forest_analysis(X, y)



signal_csv_paths = [
    'data/signal_results_s1_1000_2000_s2_1000_3740_a1_130_a2_230_with_labeling.csv',
    'data/signal_results_s1_1000_2000_a1_130_with_labeling.csv',
    'data/signal_results_s1_1000_3500_a1_220_with_labeling.csv',
    'data/signal_results_s1_1500_2870_a1_180_with_labeling.csv'
]
gt_csv_path = 'data/ground_truth_results.csv'

if __name__ == "__main__":
    main(signal_csv_paths, gt_csv_path)
