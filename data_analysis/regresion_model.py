import pandas as pd
import statsmodels.api as sm

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def prepare_data_for_analysis(df):
    # Excluding 'abs_diff_stability' to use it as the dependent variable
    independent_vars = [col for col in df.columns if col != 'abs_diff_stability']
    X = df[independent_vars]
    y = df['abs_diff_stability']
    
    # Checking for zero variance which might indicate issues for regression analysis
    for col in independent_vars:
        if X[col].var() == 0:
            print(f"Warning: Column {col} has zero variance.")
    
    if y.var() == 0:
        print("Warning: Stability_diff has zero variance.")
    
    return X, y

def regression_analysis(X, y):
    # Add a constant term to the predictor set (common practice in regression modeling)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

def main(filepath):
    df = load_data(filepath)
    X, y = prepare_data_for_analysis(df)
    regression_analysis(X, y)

processed_data_path = 'data_processed/processed_signal_data.csv'

if __name__ == "__main__":
    main(processed_data_path)
