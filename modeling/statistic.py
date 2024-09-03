import os
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew

# Load the dataset
#dataset_path = "/home/jamalids/Documents/2D/UCRArchive_2018 (copy)/InsectEPGSmallTrain/InsectEPGSmallTrain_TEST.tsv"
dataset_path="/home/jamalids/Documents/2D/data1"
#datasets = [dataset_path]
datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if f.endswith('.tsv')]
results = []

for dataset_path in datasets:
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    ts_data1 = ts_data1.drop(ts_data1.columns[0], axis=1)
    ts_data1 = ts_data1.T
    ts_data1 = ts_data1.astype(np.float64).to_numpy().reshape(-1)
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')


    # Function to calculate significant digits precision
    def count_decimal_digits(value):
        # Ignore NaN values
        if np.isnan(value):
            return 0

        # Convert to string with full precision
        value_str = f"{value:.32f}"

        # Split into integer and decimal parts
        if '.' in value_str:
            integer_part, decimal_part = value_str.split('.')

            # Remove trailing zeros from the decimal part
            decimal_part = decimal_part.rstrip('0')

            # Count the remaining digits in the decimal part
            return len(decimal_part)
        else:
            return 0  # Integer values have zero decimal precision


    # Calculate precision
    #precision_counts = [count_decimal_digits(value) for value in ts_data1 if not np.isnan(value)]
    #max_precision = np.max(precision_counts)
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
    dataset_size = os.path.getsize(dataset_path)

    # Trend changes calculation
    def count_trend_changes(data):
        diffs = np.diff(data)
        trend_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        return trend_changes

   # trend_changes = count_trend_changes(ts_data1)

    # Prepare the data for export
    statistics = {
        "datasetname":dataset_name,
        'Maximum Value': max_value,
        'Minimum Value': min_value,
        'Average Value': avg_value,
        'Positive Count': positive_count,
        'Negative Count': negative_count,
        'Unique Values': len(repetition_counts),
        'Most Frequent Value':max(repetition_counts, key=repetition_counts.get),
        'Most Frequent Value Count': max(repetition_counts.values()),
        'Total Number of Values': total_values,
        'Dataset Size (bytes)': dataset_size,

    }

    results.append(statistics)

# Convert to DataFrame and save to CSV
statistics_df = pd.DataFrame(results)
statistics_df.to_csv("statistic.csv", index=False)
print("Statistics saved to statistic.csv")
