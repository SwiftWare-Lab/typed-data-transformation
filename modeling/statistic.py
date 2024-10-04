import os
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew

# Load the dataset
dataset_path = "/home/jamalids/Documents/2D/data1/OBS/h/"
datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if f.endswith('.tsv')]
results = []

for dataset_path in datasets:
    ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
    ts_data = ts_data.drop(ts_data.columns[0], axis=1)

    ts_data1 = ts_data.iloc[0:4000000, :]
    ts_data1 = ts_data1.T
    ts_data1 = ts_data1.astype(np.float64).to_numpy().reshape(-1)
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

    # Calculate precision (optional, commented out in your original code)
    # precision_counts = [count_decimal_digits(value) for value in ts_data1 if not np.isnan(value)]
    # max_precision = np.max(precision_counts)
    # average_precision = np.mean(precision_counts)

    # Calculate other statistics
    max_value = np.max(ts_data1)
    min_value = np.min(ts_data1)
    avg_value = np.mean(ts_data1)
    positive_count = (ts_data1 > 0).sum()
    negative_count = (ts_data1 < 0).sum()
    unique, counts = np.unique(ts_data1, return_counts=True)
    repetition_counts = dict(zip(unique, counts))
    total_values = ts_data1.size
    dataset_size = total_values*4

    # Calculate the average difference of consecutive values
    if total_values > 1:
        avg_diff = np.mean(np.abs(np.diff(ts_data1)))
    else:
        avg_diff = 0  # Handle case with less than 2 values


    # Trend changes calculation
    def count_trend_changes(data):
        diffs = np.diff(data)
        trend_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        return trend_changes


    # trend_changes = count_trend_changes(ts_data1)

    # Prepare the data for export
    statistics = {
        "datasetname": dataset_name,
        'Maximum Value': max_value,
        'Minimum Value': min_value,
        'Average Value': avg_value,
        'Positive Count': positive_count,
        'Negative Count': negative_count,
        'Unique Values': len(repetition_counts),
        'Most Frequent Value': max(repetition_counts, key=repetition_counts.get),
        'Most Frequent Value Count': max(repetition_counts.values()),
        'Total Number of Values': total_values,
        'Dataset Size (bytes)': dataset_size,
        'Average Difference of Consecutive Values': avg_diff,  # Added average difference of consecutive values
    }

    results.append(statistics)

# Convert to DataFrame and save to CSV
statistics_df = pd.DataFrame(results)
statistics_df.to_csv("statistic.csv", index=False)
print("Statistics saved to statistic.csv")
