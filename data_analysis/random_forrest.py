import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def prepare_data_for_analysis(df):
    # Excluding 'diff_stability' to use it as the dependent variable
    X = df.drop('diff_stability', axis=1)
    y = df['diff_stability']
    return X, y

def plot_target_distribution(y):
    sns.histplot(y, kde=True)
    plt.title("Target Variable Distribution")
    plt.xlabel("Absolute Stability Difference")
    plt.ylabel("Frequency")
    plt.show()

def random_forest_analysis(X, y, random_split=True):
    if random_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        split_index = int(len(X) * 0.95)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

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
    
    # Plot the feature importances, sorting by label order
    plt.figure(figsize=(10, 8))
    plt.barh(X.columns, feature_importances, color='forestgreen')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importances')
    plt.show()

def main(filepath, random_split=True):
    df = load_data(filepath)
    plot_target_distribution(df['diff_stability'])
    X, y = prepare_data_for_analysis(df)
    random_forest_analysis(X, y, random_split)

processed_data_path = 'data_processed/processed_signal_data.csv'

if __name__ == "__main__":
    main(processed_data_path, random_split=False)
