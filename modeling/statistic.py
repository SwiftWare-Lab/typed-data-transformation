import os
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew

# Load the dataset
#dataset_path = "/home/jamalids/Documents/2D/UCRArchive_2018 (copy)/InsectEPGSmallTrain/InsectEPGSmallTrain_TEST.tsv"
dataset_path ="/home/jamalids/Documents/2D/data1/hst_wfc3_ir_f32.tsv"
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
ts_data1 = ts_data1.drop(ts_data1.columns[0], axis=1)
#ts_data1 = ts_data1.iloc[0:1, :]
ts_data1=ts_data1.T
ts_data1 = ts_data1.astype(np.float32).to_numpy().reshape(-1)

# Calculate statistics
max_value = np.max(ts_data1)  # Maximum value in the dataset
min_value = np.min(ts_data1)  # Minimum value in the dataset
avg_value = ts_data1.mean()  # Average of all values in the dataset
positive_count = (ts_data1 > 0).sum()  # Count of positive numbers
negative_count = (ts_data1 < 0).sum()  # Count of negative numbers
unique, counts = np.unique(ts_data1, return_counts=True)  # Number of repetitions for each unique value
repetition_counts = dict(zip(unique, counts))

# Total number of values in the dataset
total_values = ts_data1.size

# Dataset size in bytes
dataset_size = os.path.getsize(dataset_path)

# Entropy of the dataset using histogram for float data
hist, bin_edges = np.histogram(ts_data1, bins='auto', density=True)
dataset_entropy = entropy(hist, base=2)

# Standard deviation
std_deviation = np.std(ts_data1)

# Skewness of the frequency distribution
frequency_skewness = skew(ts_data1)

# Detect trend changes
def count_trend_changes(data):
    trend_changes = 0
    diffs = np.diff(data)
    trend_changes += np.sum(np.diff(np.sign(diffs)) != 0)
    return trend_changes

trend_changes = count_trend_changes(ts_data1)

# Prepare the data for export
statistics = {
    'Maximum Value': [max_value],
    'Minimum Value': [min_value],
    'Average Value': [avg_value],
    'Positive Count': [positive_count],
    'Negative Count': [negative_count],
    'Unique Values': [len(repetition_counts)],
    'Most Frequent Value': [max(repetition_counts, key=repetition_counts.get)],
    'Most Frequent Value Count': [max(repetition_counts.values())],
    'Total Number of Values': [total_values],
    'Dataset Size (bytes)': [dataset_size],
    'Entropy': [dataset_entropy],
    'Standard Deviation': [std_deviation],
    'Skewness': [frequency_skewness],
    'Trend Changes': [trend_changes]
}

# Convert to DataFrame
statistics_df = pd.DataFrame(statistics)

# Save to CSV
statistics_df.to_csv("statistic.csv", index=False)

print("Statistics saved to statistic.csv")

#################################################################

# Detect consecutive same values and count them
consecutive_counts = []
current_value = ts_data1[0]
count = 1

for i in range(1, len(ts_data1)):
    if ts_data1[i] == current_value:
        count += 1
    else:
        consecutive_counts.append((current_value, count))
        current_value = ts_data1[i]
        count = 1

# Append the last group
consecutive_counts.append((current_value, count))

# Convert to DataFrame
consecutive_df = pd.DataFrame(consecutive_counts, columns=['Value', 'Consecutive Count'])

# Save to a new CSV file
consecutive_df.to_csv("consecutive_values.csv", index=False)

print("Consecutive values saved to consecutive_values.csv")
