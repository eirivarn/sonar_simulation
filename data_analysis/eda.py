import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Path to the CSV files
path_gt = 'data/generated/elevated_seasfloor/s1_3000_1000_s2_3000_5000_a1_145_a2_215/generated_ground_truth_results.csv'  # Update with the actual path to your ground truth CSV file
path_est = 'data/generated/elevated_seasfloor/s1_3000_1000_s2_3000_5000_a1_145_a2_215/generated_signal_results.csv_s1_3000_1000_s2_3000_5000_a1_145_a2_215.csv'    # Update with the actual path to your estimates CSV file

# Load data from CSV files
df_gt = pd.read_csv(path_gt)
df_est = pd.read_csv(path_est)

# Assuming both dataframes have a common identifier or can be aligned by index
# We will perform a comparison for the 'stability_percentage' as an example

# Ensure both dataframes are sorted or aligned properly, if necessary
df_gt.sort_index(inplace=True)
df_est.sort_index(inplace=True)

if len(df_gt) == len(df_est):
    # Add estimated stability percentage to ground truth dataframe
    df_gt['est_stability_percentage'] = df_est['stability_percentage']
    df_gt['difference'] = df_gt['stability_percentage'] - df_gt['est_stability_percentage']

    # Calculate error metrics
    mae = mean_absolute_error(df_gt['stability_percentage'], df_gt['est_stability_percentage'])
    rmse = np.sqrt(mean_squared_error(df_gt['stability_percentage'], df_gt['est_stability_percentage']))
    # MAPE might have issues if there are zeros in the ground truth data, hence the use of a small constant epsilon
    epsilon = 1e-10  # Prevent division by zero
    mape = np.mean(np.abs((df_gt['stability_percentage'] - df_gt['est_stability_percentage']) / 
                          (df_gt['stability_percentage'] + epsilon))) * 100
    
    print("Mean Absolute Error (MAE): {:.2f}".format(mae))
    print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
    print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))

else:
    print("Error: Dataframes do not match in size or order.")
