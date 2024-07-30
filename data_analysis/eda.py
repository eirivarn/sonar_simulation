import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(filepath)

def calculate_correlations(df):
    """
    Calculate and plot the correlation matrix for label columns and stability percentage.
    """
    # Select only label columns plus the stability percentage
    label_columns = [col for col in df.columns if 'label_' in col]
    columns_of_interest = label_columns + ['stability_percentage']
    correlation_matrix = df[columns_of_interest].corr()
    
    # Plotting the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Labels and Stability Percentage')
    plt.show()
    return correlation_matrix

def main():
    # Define the path to your dataset
    data_path = 'data_processed/processed_signal_data.csv'
    
    # Load the data
    df = load_data(data_path)
    
    # Calculate correlations between labels and stability percentage
    corr_matrix = calculate_correlations(df)

if __name__ == "__main__":
    main()
