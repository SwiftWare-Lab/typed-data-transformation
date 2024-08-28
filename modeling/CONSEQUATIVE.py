import os
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew
dataset_path = "/home/jamalids/Documents/2D/UCRArchive_2018 (copy)/CricketX/CricketX_TEST.tsv"
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
ts_data1 = ts_data1.drop(ts_data1.columns[0], axis=1)
ts_data1 = ts_data1.astype(np.float32).to_numpy().reshape(-1)
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
#################################################
